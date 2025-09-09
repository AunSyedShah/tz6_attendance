"""
AI Attendance System - Streamlit Recognition Testing
Web-based interface for testing the enrolled student recognition capability
"""

import streamlit as st
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import Image
import io
import time

class StreamlitRecognitionTester:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.image_size = (128, 128)
        self.model = None
        self.label_encoder = None
        self.enrolled_students = []
        
    def load_enrollment_data(self):
        """Load all enrolled student data and train recognition model"""
        enrollment_dir = "enrollment_data"
        
        if not os.path.exists(enrollment_dir):
            st.error("No enrollment data found. Please enroll students first.")
            return False
        
        faces_data = []
        labels = []
        
        # Load all student directories
        with st.spinner("Loading enrollment data..."):
            for student_folder in os.listdir(enrollment_dir):
                student_path = os.path.join(enrollment_dir, student_folder)
                if os.path.isdir(student_path):
                    student_name = student_folder.split('_', 1)[1] if '_' in student_folder else student_folder
                    self.enrolled_students.append(student_name)
                    
                    st.write(f"Loading data for: {student_name}")
                    
                    # Load all images for this student
                    for image_file in os.listdir(student_path):
                        if image_file.endswith('.jpg'):
                            image_path = os.path.join(student_path, image_file)
                            face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                            
                            if face_image is not None:
                                # Flatten the image for ML model
                                face_vector = face_image.flatten()
                                faces_data.append(face_vector)
                                labels.append(student_name)
        
        if len(faces_data) == 0:
            st.error("No valid face data found.")
            return False
        
        # Prepare data for training
        X = np.array(faces_data)
        y = np.array(labels)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train KNN classifier
        self.model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        self.model.fit(X, y_encoded)
        
        st.success(f"Model trained on {len(X)} images from {len(self.enrolled_students)} students")
        st.info(f"Enrolled students: {', '.join(self.enrolled_students)}")
        
        return True
    
    def preprocess_face_for_recognition(self, frame, face_coords):
        """Same preprocessing as enrollment for consistency"""
        x, y, w, h = face_coords
        
        # Extract face with padding
        padding = int(0.2 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        face_region = gray[y1:y2, x1:x2]
        
        # Resize and enhance
        face_resized = cv2.resize(face_region, self.image_size)
        face_enhanced = cv2.equalizeHist(face_resized)
        face_final = cv2.GaussianBlur(face_enhanced, (3, 3), 0)
        
        return face_final
    
    def recognize_face(self, face_image):
        """Recognize a face using the trained model"""
        if self.model is None:
            return "Unknown", 0.0
        
        # Flatten image for prediction
        face_vector = face_image.flatten().reshape(1, -1)
        
        # Get prediction and confidence
        prediction = self.model.predict(face_vector)[0]
        probabilities = self.model.predict_proba(face_vector)[0]
        confidence = max(probabilities)
        
        # Decode label
        student_name = self.label_encoder.inverse_transform([prediction])[0]
        
        # Set confidence threshold
        if confidence < 0.6:  # Adjust threshold as needed
            return "Unknown", confidence
        
        return student_name, confidence
    
    def process_uploaded_image(self, uploaded_file):
        """Process an uploaded image for recognition"""
        if uploaded_file is not None:
            # Convert uploaded file to opencv format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Detect faces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            results = []
            processed_image = image.copy()
            
            for (x, y, w, h) in faces:
                # Preprocess face
                face_image = self.preprocess_face_for_recognition(image, (x, y, w, h))
                
                # Recognize face
                student_name, confidence = self.recognize_face(face_image)
                
                # Draw results on image
                color = (0, 255, 0) if student_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(processed_image, (x, y), (x+w, y+h), color, 2)
                
                # Display name and confidence
                label = f"{student_name} ({confidence:.2f})"
                cv2.putText(processed_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                results.append((student_name, confidence))
            
            return processed_image, results
        
        return None, []

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="AI Attendance Recognition Test",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 AI Attendance System - Recognition Test")
    st.markdown("---")
    
    # Initialize session state
    if 'tester' not in st.session_state:
        st.session_state.tester = StreamlitRecognitionTester()
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'recognition_results' not in st.session_state:
        st.session_state.recognition_results = []
    
    # Sidebar for model loading
    with st.sidebar:
        st.header("📊 Model Status")
        
        if st.button("🔄 Load Enrollment Data", type="primary"):
            st.session_state.model_loaded = st.session_state.tester.load_enrollment_data()
        
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
            st.write(f"**Enrolled Students:** {len(st.session_state.tester.enrolled_students)}")
            for student in st.session_state.tester.enrolled_students:
                st.write(f"• {student}")
        else:
            st.warning("⚠️ Model Not Loaded")
        
        st.markdown("---")
        st.header("🔧 Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📸 Recognition Testing")
        
        if not st.session_state.model_loaded:
            st.warning("Please load enrollment data first using the sidebar.")
            return
        
        # File upload option
        st.subheader("Upload Image for Recognition")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing faces to test recognition"
        )
        
        if uploaded_file is not None:
            # Process the uploaded image
            processed_image, results = st.session_state.tester.process_uploaded_image(uploaded_file)
            
            if processed_image is not None:
                # Display original and processed images
                col_orig, col_proc = st.columns(2)
                
                with col_orig:
                    st.subheader("Original Image")
                    # Convert opencv image to PIL for streamlit
                    original_rgb = cv2.cvtColor(cv2.imdecode(np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    st.image(original_rgb, use_column_width=True)
                
                with col_proc:
                    st.subheader("Recognition Results")
                    # Convert processed image to RGB for display
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, use_column_width=True)
                
                # Display recognition results
                if results:
                    st.subheader("🎯 Recognition Details")
                    for i, (name, confidence) in enumerate(results, 1):
                        confidence_color = "green" if confidence >= confidence_threshold else "red"
                        status = "✅ Recognized" if confidence >= confidence_threshold else "❌ Below Threshold"
                        
                        st.markdown(f"""
                        **Face {i}:** 
                        - **Name:** {name}
                        - **Confidence:** <span style="color:{confidence_color}">{confidence:.3f}</span>
                        - **Status:** {status}
                        """, unsafe_allow_html=True)
                        
                        # Add to results history
                        if st.button(f"📝 Record Result {i}", key=f"record_{i}"):
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.recognition_results.append({
                                'timestamp': timestamp,
                                'name': name,
                                'confidence': confidence,
                                'status': 'Recognized' if confidence >= confidence_threshold else 'Below Threshold'
                            })
                            st.success(f"Recorded: {name} with confidence {confidence:.3f}")
                else:
                    st.warning("No faces detected in the image.")
        
        # Camera capture option (requires additional setup)
        st.subheader("📹 Live Camera Recognition")
        st.info("For live camera recognition, you would need to set up camera access in your deployment environment.")
        
        # Test with sample images from enrollment data
        st.subheader("🧪 Test with Enrollment Samples")
        if st.button("Test with Random Enrollment Image"):
            enrollment_dir = "enrollment_data"
            if os.path.exists(enrollment_dir):
                # Get a random enrolled student image
                all_images = []
                for student_folder in os.listdir(enrollment_dir):
                    student_path = os.path.join(enrollment_dir, student_folder)
                    if os.path.isdir(student_path):
                        for img_file in os.listdir(student_path):
                            if img_file.endswith('.jpg'):
                                all_images.append(os.path.join(student_path, img_file))
                
                if all_images:
                    import random
                    random_image_path = random.choice(all_images)
                    test_image = cv2.imread(random_image_path)
                    
                    if test_image is not None:
                        # Process the test image
                        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                        faces = st.session_state.tester.face_cascade.detectMultiScale(gray, 1.1, 4)
                        
                        if len(faces) > 0:
                            x, y, w, h = faces[0]  # Use first face
                            face_image = st.session_state.tester.preprocess_face_for_recognition(test_image, (x, y, w, h))
                            student_name, confidence = st.session_state.tester.recognize_face(face_image)
                            
                            # Display result
                            test_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                            st.image(test_rgb, caption=f"Test Result: {student_name} (Confidence: {confidence:.3f})", use_column_width=True)
                        else:
                            st.warning("No face detected in the sample image.")
    
    with col2:
        st.header("📋 Results History")
        
        if st.session_state.recognition_results:
            st.subheader(f"Total Records: {len(st.session_state.recognition_results)}")
            
            # Display recent results
            for i, result in enumerate(reversed(st.session_state.recognition_results[-10:]), 1):
                with st.expander(f"Record {len(st.session_state.recognition_results) - i + 1}"):
                    st.write(f"**Time:** {result['timestamp']}")
                    st.write(f"**Name:** {result['name']}")
                    st.write(f"**Confidence:** {result['confidence']:.3f}")
                    st.write(f"**Status:** {result['status']}")
            
            # Export results
            if st.button("📤 Export Results"):
                import pandas as pd
                df = pd.DataFrame(st.session_state.recognition_results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"recognition_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Clear results
            if st.button("🗑️ Clear Results"):
                st.session_state.recognition_results = []
                st.success("Results cleared!")
        else:
            st.info("No recognition results yet. Upload an image to start testing!")
    
    # Footer
    st.markdown("---")
    st.markdown("**AI Attendance System** - Recognition Testing Interface")

if __name__ == "__main__":
    main()
