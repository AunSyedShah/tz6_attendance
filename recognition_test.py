"""
AI Attendance System - Recognition Testing
Tests the enrolled student recognition capability
"""

import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

class RecognitionTester:
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
            print("No enrollment data found. Please enroll students first.")
            return False
        
        faces_data = []
        labels = []
        
        # Load all student directories
        for student_folder in os.listdir(enrollment_dir):
            student_path = os.path.join(enrollment_dir, student_folder)
            if os.path.isdir(student_path):
                student_name = student_folder.split('_', 1)[1] if '_' in student_folder else student_folder
                self.enrolled_students.append(student_name)
                
                print(f"Loading data for: {student_name}")
                
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
            print("No valid face data found.")
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
        
        print(f"Model trained on {len(X)} images from {len(self.enrolled_students)} students")
        print(f"Enrolled students: {', '.join(self.enrolled_students)}")
        
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
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    
    def test_recognition(self):
        """Test the recognition system with live camera feed"""
        if not self.load_enrollment_data():
            return
        
        print("\n=== TESTING RECOGNITION SYSTEM ===")
        print("Press 'q' to quit, 'r' to record recognition result")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        recognition_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Preprocess face
                face_image = self.preprocess_face_for_recognition(frame, (x, y, w, h))
                
                # Recognize face
                student_name, confidence = self.recognize_face(face_image)
                
                # Draw results
                color = (0, 255, 0) if student_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display name and confidence
                label = f"{student_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'r' to record result, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Recognition Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and len(faces) > 0:
                # Record the recognition result
                for (x, y, w, h) in faces:
                    face_image = self.preprocess_face_for_recognition(frame, (x, y, w, h))
                    student_name, confidence = self.recognize_face(face_image)
                    recognition_results.append((student_name, confidence))
                    print(f"Recorded: {student_name} with confidence {confidence:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Display test results
        print("\n=== RECOGNITION TEST RESULTS ===")
        if recognition_results:
            for i, (name, conf) in enumerate(recognition_results, 1):
                print(f"Test {i}: {name} (Confidence: {conf:.2f})")
        else:
            print("No recognition results recorded.")

def main():
    """Demo the recognition testing"""
    tester = RecognitionTester()
    tester.test_recognition()

if __name__ == "__main__":
    main()
