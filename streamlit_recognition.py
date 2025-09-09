"""
AI Attendance System - Streamlit Recognition Interface
Streamlit-based face recognition system for enrolled students
"""

import streamlit as st
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import tempfile
from PIL import Image
import time

class StreamlitRecognitionSystem:
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
        for student_folder in os.listdir(enrollment_dir):
            student_path = os.path.join(enrollment_dir, student_folder)
            if os.path.isdir(student_path):
                student_name = student_folder.split('_', 1)[1] if '_' in student_folder else student_folder
                self.enrolled_students.append(student_name)
                
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
    
    def preprocess_face_for_recognition(self, image, face_coords):
        """Same preprocessing as enrollment for consistency"""
        x, y, w, h = face_coords
        
        # Extract face with padding
        padding = int(0.2 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
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
    
    def detect_and_recognize_faces(self, image):
        """Detect faces in image and return recognition results"""
        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        annotated_image = image.copy()
        
        for (x, y, w, h) in faces:
            # Preprocess face
            face_image = self.preprocess_face_for_recognition(image, (x, y, w, h))
            
            # Recognize face
            student_name, confidence = self.recognize_face(face_image)
            
            # Draw results on image
            color = (0, 255, 0) if student_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
            
            # Display name and confidence
            label = f"{student_name} ({confidence:.2f})"
            cv2.putText(annotated_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            results.append({
                'student_name': student_name,
                'confidence': confidence,
                'bbox': (x, y, w, h)
            })
        
        # Convert back to RGB for display
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return results, annotated_image

def main():
    st.set_page_config(
        page_title="AI Attendance System - Recognition",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("🎓 AI Attendance System - Student Recognition")
    st.markdown("Upload an image or use your camera to test student recognition")
    
    # Initialize recognition system
    if 'recognition_system' not in st.session_state:
        st.session_state.recognition_system = StreamlitRecognitionSystem()
        st.session_state.model_loaded = False
    
    recognition_system = st.session_state.recognition_system
    
    # Load model if not already loaded
    if not st.session_state.model_loaded:
        with st.spinner("Loading enrollment data and training model..."):
            if recognition_system.load_enrollment_data():
                st.session_state.model_loaded = True
            else:
                st.stop()
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📷 Camera", "📁 Upload Image", "📊 Recognition History"])
    
    with tab1:
        st.subheader("Camera Recognition")
        
        # Camera input
        camera_image = st.camera_input("Take a photo for recognition")
        
        if camera_image is not None:
            # Process the camera image
            image = Image.open(camera_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Recognition Results")
                
                with st.spinner("Processing image..."):
                    results, annotated_image = recognition_system.detect_and_recognize_faces(image)
                
                st.image(annotated_image, use_container_width=True)
                
                # Display results
                if results:
                    st.success(f"Found {len(results)} face(s)")
                    for i, result in enumerate(results, 1):
                        name = result['student_name']
                        confidence = result['confidence']
                        
                        if name != "Unknown":
                            st.write(f"**Student {i}:** {name}")
                            st.write(f"**Confidence:** {confidence:.2%}")
                            
                            # Save to session state for history
                            if 'recognition_history' not in st.session_state:
                                st.session_state.recognition_history = []
                            
                            st.session_state.recognition_history.append({
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'student_name': name,
                                'confidence': confidence,
                                'method': 'Camera'
                            })
                        else:
                            st.warning(f"**Face {i}:** Unrecognized student (Confidence: {confidence:.2%})")
                else:
                    st.warning("No faces detected in the image")
    
    with tab2:
        st.subheader("Upload Image Recognition")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing faces to recognize"
        )
        
        if uploaded_file is not None:
            # Process uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Recognition Results")
                
                with st.spinner("Processing image..."):
                    results, annotated_image = recognition_system.detect_and_recognize_faces(image)
                
                st.image(annotated_image, use_container_width=True)
                
                # Display results
                if results:
                    st.success(f"Found {len(results)} face(s)")
                    for i, result in enumerate(results, 1):
                        name = result['student_name']
                        confidence = result['confidence']
                        
                        if name != "Unknown":
                            st.write(f"**Student {i}:** {name}")
                            st.write(f"**Confidence:** {confidence:.2%}")
                            
                            # Save to session state for history
                            if 'recognition_history' not in st.session_state:
                                st.session_state.recognition_history = []
                            
                            st.session_state.recognition_history.append({
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'student_name': name,
                                'confidence': confidence,
                                'method': 'Upload'
                            })
                        else:
                            st.warning(f"**Face {i}:** Unrecognized student (Confidence: {confidence:.2%})")
                else:
                    st.warning("No faces detected in the image")
    
    with tab3:
        st.subheader("Recognition History")
        
        if 'recognition_history' in st.session_state and st.session_state.recognition_history:
            import pandas as pd
            
            df = pd.DataFrame(st.session_state.recognition_history)
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Recognitions", len(df))
            
            with col2:
                avg_confidence = df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.2%}")
            
            with col3:
                unique_students = df['student_name'].nunique()
                st.metric("Unique Students", unique_students)
            
            # Clear history button
            if st.button("Clear History", type="secondary"):
                st.session_state.recognition_history = []
                st.rerun()
        else:
            st.info("No recognition history yet. Use the camera or upload images to start recognizing students.")
    
    # Sidebar with system info
    with st.sidebar:
        st.subheader("System Information")
        
        if st.session_state.model_loaded:
            st.success("✅ Model loaded successfully")
            st.write(f"**Enrolled Students:** {len(recognition_system.enrolled_students)}")
            
            with st.expander("View Enrolled Students"):
                for student in recognition_system.enrolled_students:
                    st.write(f"• {student}")
        else:
            st.error("❌ Model not loaded")
        
        st.subheader("Recognition Settings")
        
        # Add confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6, 
            step=0.1,
            help="Minimum confidence required for positive identification"
        )
        
        st.subheader("About")
        st.markdown("""
        This AI Attendance System uses:
        - **Face Detection**: Haar Cascades
        - **Recognition**: K-Nearest Neighbors
        - **Features**: Preprocessed face images
        - **Interface**: Streamlit web app
        """)

if __name__ == "__main__":
    main()
