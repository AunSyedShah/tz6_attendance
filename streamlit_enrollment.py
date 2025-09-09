"""
AI Attendance System - Streamlit Enrollment Interface
Streamlit-based student enrollment system with image preprocessing
"""

import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime
import time
from PIL import Image
import tempfile

class StreamlitEnrollmentSystem:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Enrollment settings
        self.target_images_per_session = 20  # Total images to capture
        self.face_size_threshold = (50, 50)  # Minimum face size
        self.image_size = (128, 128)  # Standard face image size
        
        # Quality thresholds
        self.blur_threshold = 100  # Laplacian variance threshold for blur detection
        self.brightness_range = (50, 200)  # Acceptable brightness range
        
    def detect_and_validate_face(self, image):
        """
        Face Detection with Quality Validation
        Returns: (face_detected, face_region, quality_status)
        """
        # Convert PIL image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return False, None, "No face detected"
        elif len(faces) > 1:
            return False, None, "Multiple faces detected - please ensure only one person"
        
        # Get the single detected face
        (x, y, w, h) = faces[0]
        
        # Quality checks
        face_region = gray[y:y+h, x:x+w]
        
        # 1. Size validation
        if w < self.face_size_threshold[0] or h < self.face_size_threshold[1]:
            return False, (x, y, w, h), "Face too small - move closer to camera"
        
        # 2. Position validation (face should be roughly centered)
        frame_center_x = image.shape[1] // 2
        face_center_x = x + w // 2
        if abs(face_center_x - frame_center_x) > image.shape[1] * 0.25:
            return False, (x, y, w, h), "Please center your face in the frame"
        
        # 3. Blur detection using Laplacian variance
        blur_value = cv2.Laplacian(face_region, cv2.CV_64F).var()
        if blur_value < self.blur_threshold:
            return False, (x, y, w, h), "Image too blurry - hold camera steady"
        
        # 4. Brightness validation
        brightness = np.mean(face_region)
        if brightness < self.brightness_range[0]:
            return False, (x, y, w, h), "Too dark - improve lighting"
        elif brightness > self.brightness_range[1]:
            return False, (x, y, w, h), "Too bright - reduce lighting"
        
        return True, (x, y, w, h), "Good quality - ready to capture"
    
    def preprocess_face_image(self, image, face_coords):
        """
        Image Pre-processing - Extract and enhance face region for training
        """
        x, y, w, h = face_coords
        
        # Convert to proper format if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Extract face region with padding
        padding = int(0.2 * max(w, h))  # 20% padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # 2. Convert to grayscale and extract face
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_region = gray[y1:y2, x1:x2]
        
        # 3. Resize to standard dimensions
        face_resized = cv2.resize(face_region, self.image_size)
        
        # 4. Apply histogram equalization for better contrast
        face_enhanced = cv2.equalizeHist(face_resized)
        
        # 5. Apply slight Gaussian blur to reduce noise
        face_final = cv2.GaussianBlur(face_enhanced, (3, 3), 0)
        
        return face_final
    
    def create_student_directory(self, student_name, student_id):
        """Create directory structure for student data"""
        base_dir = "enrollment_data"
        # Clean student name for directory (remove special characters)
        clean_name = "".join(c for c in student_name if c.isalnum() or c in (' ', '-', '_')).strip()
        student_dir = f"{base_dir}/{student_id}_{clean_name}"
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(student_dir):
            os.makedirs(student_dir)
            
        return student_dir
    
    def process_and_save_image(self, image, student_dir, image_count):
        """Process uploaded image and save if face is detected and valid"""
        face_detected, face_coords, status = self.detect_and_validate_face(image)
        
        if face_detected and face_coords:
            # Preprocess the face
            processed_face = self.preprocess_face_image(image, face_coords)
            
            # Save processed image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{student_dir}/face_{image_count:03d}_{timestamp}.jpg"
            cv2.imwrite(filename, processed_face)
            
            return True, status, processed_face
        
        return False, status, None

def main():
    st.set_page_config(
        page_title="AI Attendance System - Enrollment",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 AI Attendance System - Student Enrollment")
    st.markdown("Create dataset for new students with proper image preprocessing")
    
    # Initialize enrollment system
    if 'enrollment_system' not in st.session_state:
        st.session_state.enrollment_system = StreamlitEnrollmentSystem()
    
    enrollment_system = st.session_state.enrollment_system
    
    # Main enrollment interface
    st.subheader("Student Information")
    
    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("Student Name", placeholder="Enter full name")
    with col2:
        student_id = st.text_input("Student ID", placeholder="Enter student ID/number")
    
    if not student_name or not student_id:
        st.warning("Please enter both student name and ID to continue.")
        st.stop()
    
    # Initialize session state for this student
    session_key = f"{student_id}_{student_name}"
    if 'current_student' not in st.session_state or st.session_state.current_student != session_key:
        st.session_state.current_student = session_key
        st.session_state.captured_images = 0
        st.session_state.student_dir = enrollment_system.create_student_directory(student_name, student_id)
    
    # Display current progress
    st.subheader("Enrollment Progress")
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    
    with progress_col1:
        st.metric("Student", student_name)
    with progress_col2:
        st.metric("ID", student_id)
    with progress_col3:
        captured = st.session_state.captured_images
        target = enrollment_system.target_images_per_session
        st.metric("Images Captured", f"{captured}/{target}")
    
    # Progress bar
    progress = captured / target if target > 0 else 0
    st.progress(progress)
    
    if captured >= target:
        st.success(f"✅ Enrollment complete! {captured} images captured for {student_name}")
        
        if st.button("Start New Enrollment", type="primary"):
            # Reset for new student
            if 'current_student' in st.session_state:
                del st.session_state.current_student
            if 'captured_images' in st.session_state:
                del st.session_state.captured_images
            st.rerun()
        
        st.stop()
    
    # Enrollment methods
    tab1, tab2, tab3 = st.tabs(["📷 Camera Capture", "📁 Upload Images", "📊 Preview Dataset"])
    
    with tab1:
        st.subheader("Camera Capture")
        st.markdown("Use your camera to capture images for enrollment")
        
        # Camera input
        camera_image = st.camera_input("Take a photo")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Captured Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Face Detection & Quality Check")
                
                # Process the image
                face_detected, face_coords, status = enrollment_system.detect_and_validate_face(image)
                
                # Show status
                if face_detected:
                    st.success(status)
                    
                    # Show detected face with bounding box
                    img_array = np.array(image)
                    if face_coords:
                        x, y, w, h = face_coords
                        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    st.image(img_array, caption="Face detected", use_container_width=True)
                    
                    # Capture button
                    if st.button("💾 Save This Image", type="primary", key="save_camera"):
                        success, msg, processed_face = enrollment_system.process_and_save_image(
                            image, st.session_state.student_dir, st.session_state.captured_images + 1
                        )
                        
                        if success:
                            st.session_state.captured_images += 1
                            st.success(f"Image saved! ({st.session_state.captured_images}/{target})")
                            
                            # Show processed face
                            if processed_face is not None:
                                st.subheader("Processed Face (saved to dataset)")
                                st.image(processed_face, width=128, caption="Preprocessed face image")
                            
                            time.sleep(1)  # Brief pause before rerun
                            st.rerun()
                        else:
                            st.error(f"Failed to save image: {msg}")
                else:
                    st.error(status)
                    
                    # Show image with any detected faces (even if invalid)
                    if face_coords:
                        img_array = np.array(image)
                        x, y, w, h = face_coords
                        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        st.image(img_array, caption="Issues detected", use_container_width=True)
    
    with tab2:
        st.subheader("Upload Multiple Images")
        st.markdown("Upload multiple images at once for batch processing")
        
        uploaded_files = st.file_uploader(
            "Choose image files", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple images containing the student's face"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            if st.button("Process All Images", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                successful_saves = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    try:
                        image = Image.open(uploaded_file)
                        
                        success, msg, processed_face = enrollment_system.process_and_save_image(
                            image, st.session_state.student_dir, st.session_state.captured_images + successful_saves + 1
                        )
                        
                        if success:
                            successful_saves += 1
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                # Update session state
                st.session_state.captured_images += successful_saves
                
                status_text.text("Processing complete!")
                st.success(f"Successfully processed {successful_saves} out of {len(uploaded_files)} images")
                
                if successful_saves > 0:
                    time.sleep(2)
                    st.rerun()
    
    with tab3:
        st.subheader("Dataset Preview")
        
        if st.session_state.captured_images > 0:
            st.write(f"**Dataset Location:** `{st.session_state.student_dir}`")
            
            # Load and display some sample images
            image_files = [f for f in os.listdir(st.session_state.student_dir) if f.endswith('.jpg')]
            
            if image_files:
                st.write(f"**Total Images:** {len(image_files)}")
                
                # Show sample images
                st.subheader("Sample Processed Images")
                cols = st.columns(min(5, len(image_files)))
                
                for i, img_file in enumerate(image_files[:5]):
                    with cols[i]:
                        img_path = os.path.join(st.session_state.student_dir, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            st.image(img, caption=f"Image {i+1}", width=100)
                
                # Dataset statistics
                if len(image_files) > 0:
                    sample_img = cv2.imread(os.path.join(st.session_state.student_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)
                    if sample_img is not None:
                        st.subheader("Dataset Statistics")
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        
                        with stat_col1:
                            st.metric("Image Count", len(image_files))
                        with stat_col2:
                            st.metric("Image Size", f"{sample_img.shape[0]}x{sample_img.shape[1]}")
                        with stat_col3:
                            completion = (len(image_files) / target) * 100
                            st.metric("Completion", f"{completion:.1f}%")
        else:
            st.info("No images captured yet. Use the Camera or Upload tabs to add images to the dataset.")
    
    # Sidebar with instructions and settings
    with st.sidebar:
        st.subheader("📋 Enrollment Guide")
        
        st.markdown("""
        **For Best Results:**
        1. **Good Lighting**: Ensure face is well-lit
        2. **Center Face**: Keep face centered in frame
        3. **Stay Still**: Avoid motion blur
        4. **One Person**: Only one face per image
        5. **Varied Poses**: Slight variations in angle/expression
        """)
        
        st.subheader("⚙️ Quality Settings")
        
        # Display current thresholds
        st.write(f"**Blur Threshold:** {enrollment_system.blur_threshold}")
        st.write(f"**Brightness Range:** {enrollment_system.brightness_range[0]}-{enrollment_system.brightness_range[1]}")
        st.write(f"**Target Images:** {enrollment_system.target_images_per_session}")
        st.write(f"**Image Size:** {enrollment_system.image_size[0]}x{enrollment_system.image_size[1]}")
        
        st.subheader("📁 Dataset Info")
        
        # Show existing enrolled students
        if os.path.exists("enrollment_data"):
            existing_students = [d for d in os.listdir("enrollment_data") if os.path.isdir(os.path.join("enrollment_data", d))]
            if existing_students:
                st.write("**Enrolled Students:**")
                for student in existing_students:
                    student_name_display = student.split('_', 1)[1] if '_' in student else student
                    st.write(f"• {student_name_display}")
            else:
                st.write("No students enrolled yet.")
        else:
            st.write("No enrollment data directory found.")

if __name__ == "__main__":
    main()
