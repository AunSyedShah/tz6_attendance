"""
AI Attendance System - Student Enrollment Module
Streamlit application for student enrollment with interactive prompts and image capture
Supports both live camera capture and file upload for testing in environments without camera access
"""

import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime
import time
from PIL import Image
import tempfile
import threading
import queue

class StudentEnrollmentSystem:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Enrollment settings
        self.target_images_total = 100  # Total images to capture per student
        self.images_per_prompt = 15     # Images per pose/prompt
        self.face_size_threshold = (50, 50)  # Minimum face size
        self.image_size = (128, 128)    # Standard face image size
        
        # Quality thresholds for low light handling
        self.blur_threshold = 80        # Laplacian variance threshold
        self.brightness_range = (15, 230)  # Very lenient brightness range
        self.low_light_threshold = 50   # Threshold for applying enhancement
        
        # Interactive prompts for comprehensive face capture
        self.enrollment_prompts = [
            "Look straight ahead at the camera",
            "Look slightly up", 
            "Look slightly down",
            "Turn your head slightly to the left",
            "Turn your head slightly to the right",
            "Tilt your head slightly to the left",
            "Tilt your head slightly to the right",
            "Smile naturally",
            "Neutral expression - no smile",
            "Look straight with mouth slightly open"
        ]
        
    def check_camera_availability(self):
        """Check if camera is available and accessible with multiple backends"""
        try:
            # Try different camera indices and backends
            backends = [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_GSTREAMER]
            
            for backend in backends:
                for camera_index in range(5):  # Try indices 0-4
                    try:
                        cap = cv2.VideoCapture(camera_index, backend)
                        if cap.isOpened():
                            # Test actual frame capture
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                cap.release()
                                backend_name = {
                                    cv2.CAP_ANY: "ANY",
                                    cv2.CAP_V4L2: "V4L2", 
                                    cv2.CAP_DSHOW: "DSHOW",
                                    cv2.CAP_GSTREAMER: "GSTREAMER"
                                }.get(backend, "UNKNOWN")
                                return True, f"Camera found at index {camera_index} using {backend_name}"
                            cap.release()
                    except Exception as e:
                        continue
            
            return False, "No accessible camera found. Please check camera connection and permissions."
        except Exception as e:
            return False, f"Camera detection error: {str(e)}"
        
    def setup_camera_for_optimal_capture(self, cap):
        """Optimize camera settings for best capture quality"""
        try:
            # Camera optimization settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
            cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
        except Exception as e:
            st.warning(f"Could not optimize camera settings: {e}")
        return cap
    
    def get_camera_debug_info(self):
        """Get detailed camera debug information"""
        debug_info = []
        
        try:
            import platform
            debug_info.append(f"Platform: {platform.system()} {platform.release()}")
            
            # Check OpenCV version
            debug_info.append(f"OpenCV Version: {cv2.__version__}")
            
            # Check available camera backends
            backends = []
            if hasattr(cv2, 'CAP_V4L2'):
                backends.append("V4L2")
            if hasattr(cv2, 'CAP_DSHOW'):
                backends.append("DirectShow")
            if hasattr(cv2, 'CAP_GSTREAMER'):
                backends.append("GStreamer")
            debug_info.append(f"Available backends: {', '.join(backends)}")
            
            # Try to detect camera indices
            working_indices = []
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        working_indices.append(str(i))
                cap.release()
            
            debug_info.append(f"Detected camera indices: {', '.join(working_indices) if working_indices else 'None'}")
            
        except Exception as e:
            debug_info.append(f"Debug error: {str(e)}")
        
        return debug_info
        """Get a working camera instance"""
        backends = [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_GSTREAMER]
        
        for backend in backends:
            for camera_index in range(5):
                try:
                    cap = cv2.VideoCapture(camera_index, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            # Reset camera to beginning
                            cap.release()
                            cap = cv2.VideoCapture(camera_index, backend)
                            return self.setup_camera_for_optimal_capture(cap)
                        cap.release()
                except Exception:
                    continue
        return None
    
    def detect_and_validate_face(self, frame):
        """
        Detect face and validate quality with enhanced low-light support
        Returns: (face_detected, face_coords, status_message, confidence_score)
        """
        if frame is None:
            return False, None, "No frame received", 0
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply enhancement for low light conditions
        brightness = np.mean(gray)
        if brightness < self.low_light_threshold:
            # Apply CLAHE for better detection in low light
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        
        # Detect faces with multiple scale factors for better detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=self.face_size_threshold,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return False, None, "❌ No face detected - please position yourself in front of camera", 0
        elif len(faces) > 1:
            return False, None, "❌ Multiple faces detected - ensure only one person is visible", 0
        
        # Get the detected face
        (x, y, w, h) = faces[0]
        face_region = gray[y:y+h, x:x+w]
        
        # Quality validations
        confidence_score = 0
        
        # 1. Size validation
        if w < self.face_size_threshold[0] or h < self.face_size_threshold[1]:
            return False, (x, y, w, h), "❌ Face too small - move closer to camera", 10
        confidence_score += 20
        
        # 2. Position validation (face should be reasonably centered)
        frame_center_x = frame.shape[1] // 2
        face_center_x = x + w // 2
        if abs(face_center_x - frame_center_x) > frame.shape[1] * 0.3:
            return False, (x, y, w, h), "❌ Please center your face in the frame", 30
        confidence_score += 20
        
        # 3. Blur detection with more lenient threshold
        blur_value = cv2.Laplacian(face_region, cv2.CV_64F).var()
        if blur_value < self.blur_threshold:
            return False, (x, y, w, h), "❌ Image too blurry - please hold still", 50
        confidence_score += 30
        
        # 4. Enhanced brightness validation
        face_brightness = np.mean(face_region)
        if face_brightness < self.brightness_range[0]:
            return False, (x, y, w, h), "❌ Extremely low light - please improve lighting", 70
        elif face_brightness > self.brightness_range[1]:
            return False, (x, y, w, h), "❌ Too bright - please reduce lighting", 70
        confidence_score += 30
        
        # Face detected and passed all quality checks
        quality_msg = "✅ Excellent quality - capturing image!"
        if face_brightness < self.low_light_threshold:
            quality_msg = "✅ Low light detected - auto-enhancing and capturing!"
        
        return True, (x, y, w, h), quality_msg, min(100, confidence_score)
    
    def preprocess_face_image(self, frame, face_coords):
        """
        Advanced image preprocessing with low-light enhancement
        """
        x, y, w, h = face_coords
        
        # Extract face region with padding
        padding = int(0.25 * max(w, h))  # 25% padding for better context
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Convert to grayscale and extract face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_region = gray[y1:y2, x1:x2]
        
        # Resize to standard dimensions
        face_resized = cv2.resize(face_region, self.image_size)
        
        # Apply appropriate enhancement based on lighting conditions
        brightness = np.mean(face_resized)
        
        if brightness < self.low_light_threshold:
            # Advanced low-light enhancement
            # 1. CLAHE for adaptive contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            face_enhanced = clahe.apply(face_resized)
            
            # 2. Gamma correction for very dark images
            if brightness < 30:
                gamma = 1.5
                face_enhanced = np.array(255 * (face_enhanced / 255) ** (1/gamma), dtype=np.uint8)
                
            # 3. Bilateral filter to reduce noise while preserving edges
            face_enhanced = cv2.bilateralFilter(face_enhanced, 9, 75, 75)
        else:
            # Standard enhancement for normal lighting
            face_enhanced = cv2.equalizeHist(face_resized)
        
        # Final noise reduction
        face_final = cv2.GaussianBlur(face_enhanced, (3, 3), 0)
        
        return face_final
    
    def create_student_directory(self, student_name, student_id):
        """Create organized directory structure for student data"""
        base_dir = "enrollment_data"
        
        # Clean student name for directory (remove special characters)
        clean_name = "".join(c for c in student_name if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_name = clean_name.replace(' ', '_')
        
        student_dir = f"{base_dir}/{student_id}_{clean_name}"
        
        # Create directories if they don't exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(student_dir):
            os.makedirs(student_dir)
            
        return student_dir
    
    def enhance_frame_for_display(self, frame):
        """Enhance frame for better visibility during enrollment"""
        if frame is None:
            return frame
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < self.low_light_threshold:
            # Enhance for display without affecting processing
            enhanced = cv2.convertScaleAbs(frame, alpha=1.4, beta=25)
            return enhanced
        
        return frame

def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="AI Attendance System - Student Enrollment",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the enrollment system
    if 'enrollment_system' not in st.session_state:
        st.session_state.enrollment_system = StudentEnrollmentSystem()
    
    enrollment_system = st.session_state.enrollment_system
    
    # Main title and description
    st.title("🎓 AI Attendance System")
    st.header("📝 Student Enrollment Module")
    st.markdown("""
    Welcome to the student enrollment system! This module will capture comprehensive facial data
    for accurate attendance recognition. The system will guide you through various poses and
    expressions to ensure robust face recognition.
    """)
    
    # Check camera availability
    camera_available, camera_status = enrollment_system.check_camera_availability()
    
    # Display camera status
    if camera_available:
        st.success(f"📷 {camera_status}")
    else:
        st.warning(f"📷 {camera_status}")
        st.info("💡 Don't worry! You can still test the system using the file upload mode below.")
        
        # Camera troubleshooting
        with st.expander("🔧 Camera Troubleshooting"):
            st.markdown("""
            **If your camera is connected but not detected:**
            1. **Refresh the page** - Camera might need re-detection
            2. **Check permissions** - Browser needs camera access
            3. **Close other applications** using the camera
            4. **Try different browsers** (Chrome, Firefox, Edge)
            5. **Use Demo Mode** as an alternative
            """)
            
            if st.button("🔄 Re-test Camera"):
                st.session_state.enrollment_system = StudentEnrollmentSystem()
                st.rerun()
            
            if st.button("🐛 Show Debug Info"):
                debug_info = enrollment_system.get_camera_debug_info()
                for info in debug_info:
                    st.text(info)
            
            st.markdown("**Advanced: Manual Camera Selection**")
            manual_camera_index = st.number_input(
                "Camera Index", 
                min_value=0, 
                max_value=10, 
                value=0,
                help="Try different camera indices if auto-detection fails"
            )
            
            if st.button("🎯 Test Manual Camera Index"):
                try:
                    cap = cv2.VideoCapture(int(manual_camera_index))
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            st.success(f"✅ Camera {manual_camera_index} works!")
                        else:
                            st.error(f"❌ Camera {manual_camera_index} detected but can't read frames")
                        cap.release()
                    else:
                        st.error(f"❌ Camera {manual_camera_index} not accessible")
                except Exception as e:
                    st.error(f"❌ Error testing camera {manual_camera_index}: {e}")
    
    # Enrollment mode selection
    st.subheader("🔧 Enrollment Mode")
    col1, col2 = st.columns(2)
    
    with col1:
        if camera_available:
            live_mode = st.button("📷 Live Camera Enrollment", use_container_width=True, type="primary")
        else:
            live_mode = st.button("📷 Live Camera Enrollment (Unavailable)", use_container_width=True, disabled=True)
            st.caption("Camera not accessible in this environment")
    
    with col2:
        demo_mode = st.button("📁 Demo Mode (File Upload)", use_container_width=True)
        st.caption("Upload images to test the enrollment system")
    
    # Sidebar for student information
    st.sidebar.header("👤 Student Information")
    
    # Student details input
    student_name = st.sidebar.text_input(
        "Student Full Name *",
        placeholder="Enter full name (e.g., John Doe)",
        help="Enter the complete name of the student"
    )
    
    student_id = st.sidebar.text_input(
        "Student ID *",
        placeholder="Enter student ID (e.g., ST2024001)",
        help="Enter unique student identifier"
    )
    
    # Validation
    if student_name and student_id:
        st.sidebar.success("✅ Student information provided")
        
        # Check if student already exists
        enrollment_system = st.session_state.enrollment_system
        student_dir = enrollment_system.create_student_directory(student_name, student_id)
        
        if os.path.exists(student_dir) and os.listdir(student_dir):
            st.sidebar.warning(f"⚠️ Student {student_name} already enrolled!")
            if st.sidebar.button("🗑️ Re-enroll (Delete existing data)"):
                import shutil
                shutil.rmtree(student_dir)
                st.sidebar.success("Previous enrollment data deleted")
                st.rerun()
        
        # Enrollment controls
        st.sidebar.header("📸 Enrollment Controls")
        
        # Handle enrollment mode selection
        if live_mode and camera_available:
            st.session_state.enrollment_mode = "live"
            st.session_state.enrollment_active = True
            st.session_state.current_prompt_index = 0
            st.session_state.images_captured = 0
            st.session_state.prompt_images_count = 0
            st.rerun()
        elif demo_mode:
            st.session_state.enrollment_mode = "demo"
            st.session_state.enrollment_active = True
            st.rerun()
        
        if st.sidebar.button("🚀 Start Live Enrollment", type="primary", disabled=not camera_available):
            if camera_available:
                st.session_state.enrollment_mode = "live"
                st.session_state.enrollment_active = True
                st.session_state.current_prompt_index = 0
                st.session_state.images_captured = 0
                st.session_state.prompt_images_count = 0
                st.rerun()
        
        if st.sidebar.button("📁 Start Demo Mode"):
            st.session_state.enrollment_mode = "demo"
            st.session_state.enrollment_active = True
            st.rerun()

        if st.sidebar.button("⏹️ Stop Enrollment"):
            st.session_state.enrollment_active = False
            st.session_state.enrollment_mode = None
            st.rerun()
            
    else:
        st.sidebar.error("❌ Please provide both student name and ID")
    
    # Main enrollment interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📷 Live Camera Feed")
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        
        # Status and progress
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
    with col2:
        st.subheader("📊 Enrollment Progress")
        
        # Progress metrics
        if 'images_captured' not in st.session_state:
            st.session_state.images_captured = 0
        
        progress_metrics = st.empty()
        
        # Current prompt display
        prompt_display = st.empty()
        
        # Instructions
        st.markdown("""
        ### 📋 Instructions:
        1. **Position yourself** clearly in front of the camera
        2. **Follow the prompts** displayed on screen
        3. **Hold still** when the system detects your face
        4. **Wait for confirmation** before moving to next pose
        5. The system will automatically capture **100+ images**
        
        ### ✅ Quality Requirements:
        - Face should be **clearly visible**
        - Good **lighting conditions**
        - **Center your face** in the frame
        - **Hold steady** to avoid blurry images
        """)
    
    # Enrollment process
    if 'enrollment_active' in st.session_state and st.session_state.enrollment_active:
        if student_name and student_id:
            enrollment_mode = st.session_state.get('enrollment_mode', 'live')
            
            if enrollment_mode == 'live':
                run_live_enrollment_process(
                    camera_placeholder, status_placeholder, progress_placeholder,
                    progress_metrics, prompt_display, student_name, student_id
                )
            elif enrollment_mode == 'demo':
                run_demo_enrollment_process(
                    camera_placeholder, status_placeholder, progress_placeholder,
                    progress_metrics, prompt_display, student_name, student_id
                )
        else:
            st.error("Please provide student information before starting enrollment")
            st.session_state.enrollment_active = False
    
    # Footer
    st.markdown("---")
    st.markdown("**AI Attendance System** - Powered by Computer Vision & Machine Learning")

def run_live_enrollment_process(camera_placeholder, status_placeholder, progress_placeholder, 
                               progress_metrics, prompt_display, student_name, student_id):
    """Run the interactive live camera enrollment process"""
    
    enrollment_system = st.session_state.enrollment_system
    
    # Initialize session state variables
    if 'current_prompt_index' not in st.session_state:
        st.session_state.current_prompt_index = 0
    if 'images_captured' not in st.session_state:
        st.session_state.images_captured = 0
    if 'prompt_images_count' not in st.session_state:
        st.session_state.prompt_images_count = 0
    
    # Create student directory
    student_dir = enrollment_system.create_student_directory(student_name, student_id)
    
    # Current prompt
    current_prompt = enrollment_system.enrollment_prompts[st.session_state.current_prompt_index]
    
    # Update prompt display
    prompt_display.markdown(f"""
    ### 🎯 Current Instruction:
    **"{current_prompt}"**
    
    **Prompt:** {st.session_state.current_prompt_index + 1} of {len(enrollment_system.enrollment_prompts)}
    """)
    
    # Update progress metrics
    progress_metrics.metric(
        label="Total Images Captured",
        value=f"{st.session_state.images_captured}",
        delta=f"Target: {enrollment_system.target_images_total}"
    )
    
    # Progress bar
    progress = st.session_state.images_captured / enrollment_system.target_images_total
    progress_placeholder.progress(progress, text=f"Progress: {progress:.1%}")
    
    # Camera capture
    try:
        # Initialize camera with improved detection
        cap = enrollment_system.get_working_camera()
        
        if cap is None or not cap.isOpened():
            st.error("❌ Could not access camera. Please check camera connection and permissions.")
            st.error("💡 Try refreshing the page or switching to Demo Mode with file upload.")
            st.session_state.enrollment_active = False
            return
        
        # Capture and process frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # Enhance frame for display
            display_frame = enrollment_system.enhance_frame_for_display(frame)
            
            # Face detection and validation
            face_detected, face_coords, status_msg, confidence = enrollment_system.detect_and_validate_face(frame)
            
            # Draw face rectangle and status
            if face_coords:
                x, y, w, h = face_coords
                color = (0, 255, 0) if face_detected else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Add confidence score
                if face_detected:
                    cv2.putText(display_frame, f"Quality: {confidence}%", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add prompt text to frame
            cv2.putText(display_frame, current_prompt, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Images: {st.session_state.prompt_images_count}/{enrollment_system.images_per_prompt}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(display_frame_rgb, channels="RGB", use_column_width=True)
            
            # Status update
            status_placeholder.markdown(f"**Status:** {status_msg}")
            
            # Auto-capture if face is detected and validated
            if face_detected and face_coords:
                # Process and save image
                processed_face = enrollment_system.preprocess_face_image(frame, face_coords)
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                filename = f"{student_dir}/face_{st.session_state.current_prompt_index:02d}_{st.session_state.prompt_images_count:03d}_{timestamp}.jpg"
                cv2.imwrite(filename, processed_face)
                
                # Update counters
                st.session_state.images_captured += 1
                st.session_state.prompt_images_count += 1
                
                # Check if current prompt is complete
                if st.session_state.prompt_images_count >= enrollment_system.images_per_prompt:
                    st.session_state.current_prompt_index += 1
                    st.session_state.prompt_images_count = 0
                    
                    # Check if all prompts are complete
                    if st.session_state.current_prompt_index >= len(enrollment_system.enrollment_prompts):
                        # Enrollment complete
                        st.session_state.enrollment_active = False
                        st.success(f"🎉 Enrollment Complete! {st.session_state.images_captured} images captured for {student_name}")
                        st.balloons()
                        return
                
                # Auto-refresh for next capture
                time.sleep(0.2)  # Small delay between captures
                st.rerun()
        
        else:
            st.error("❌ Could not capture frame from camera")
            
    except Exception as e:
        st.error(f"❌ Camera error: {str(e)}")
        st.session_state.enrollment_active = False

def run_demo_enrollment_process(camera_placeholder, status_placeholder, progress_placeholder,
                               progress_metrics, prompt_display, student_name, student_id):
    """Run the demo enrollment process using file uploads"""
    
    enrollment_system = st.session_state.enrollment_system
    
    # Create student directory
    student_dir = enrollment_system.create_student_directory(student_name, student_id)
    
    with camera_placeholder.container():
        st.info("📁 **Demo Mode**: Upload images to simulate the enrollment process")
        
        # File uploader for multiple images
        uploaded_files = st.file_uploader(
            "Upload face images (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple clear face images from different angles"
        )
        
        if uploaded_files:
            total_uploaded = len(uploaded_files)
            successful_uploads = 0
            
            # Progress bar
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Load image
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    
                    # Convert to OpenCV format if needed
                    if len(image_array.shape) == 3:
                        if image_array.shape[2] == 3:  # RGB
                            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        else:  # RGBA
                            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                    else:
                        image_cv = image_array
                    
                    # Face detection and validation
                    face_detected, face_coords, status_msg, confidence = enrollment_system.detect_and_validate_face(image_cv)
                    
                    if face_detected and face_coords:
                        # Process and save the image
                        processed_face = enrollment_system.preprocess_face_image(image_cv, face_coords)
                        
                        # Save processed image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{student_dir}/demo_{successful_uploads:03d}_{timestamp}.jpg"
                        cv2.imwrite(filename, processed_face)
                        
                        successful_uploads += 1
                        
                        # Display the processed image
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption=f"Original: {uploaded_file.name}", width=200)
                        with col2:
                            st.image(processed_face, caption=f"Processed (Saved)", width=200)
                        
                        st.success(f"✅ {status_msg} - Image {successful_uploads} saved")
                    else:
                        st.error(f"❌ {uploaded_file.name}: {status_msg}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_uploaded)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            # Update progress metrics
            with progress_metrics.container():
                st.metric("Images Processed", f"{successful_uploads}/{total_uploaded}")
                st.metric("Success Rate", f"{(successful_uploads/total_uploaded)*100:.1f}%" if total_uploaded > 0 else "0%")
                
                if successful_uploads >= 10:
                    st.success("🎉 Sufficient images collected for basic training!")
                elif successful_uploads >= 5:
                    st.warning("⚠️ More images recommended for better accuracy")
                else:
                    st.info("📸 Upload more images for better results")
        
        else:
            st.info("Please upload face images to start the demo enrollment process")
    
    with status_placeholder.container():
        st.markdown("### 📋 Demo Mode Instructions")
        st.markdown("""
        1. **Prepare Images**: Collect clear face images from different angles
        2. **Upload Files**: Use the file uploader above to select multiple images
        3. **Review Results**: Check which images are accepted/rejected
        4. **Quality Tips**: Ensure good lighting and clear face visibility
        """)

if __name__ == "__main__":
    main()
