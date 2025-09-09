"""
AI Attendance System - Enrollment Phase Prototype
This demonstrates real-time face detection and image pre-processing during enrollment
"""

import cv2
import os
import numpy as np
from datetime import datetime
import time

class EnrollmentSystem:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Enrollment settings
        self.target_images_per_pose = 15  # Images per pose/prompt
        self.face_size_threshold = (50, 50)  # Minimum face size
        self.image_size = (128, 128)  # Standard face image size
        
        # Quality thresholds
        self.blur_threshold = 100  # Laplacian variance threshold for blur detection
        self.brightness_range = (50, 200)  # Acceptable brightness range
        
        # Enrollment prompts
        self.enrollment_prompts = [
            "Look straight ahead",
            "Look slightly up",
            "Look slightly down", 
            "Turn head slightly left",
            "Turn head slightly right",
            "Slight smile please",
            "Neutral expression"
        ]
        
    def detect_and_validate_face(self, frame):
        """
        Step 2: Face Detection with Quality Validation
        Returns: (face_detected, face_region, quality_status)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return False, None, "No face detected"
        elif len(faces) > 1:
            return False, None, "Multiple faces detected"
        
        # Get the single detected face
        (x, y, w, h) = faces[0]
        
        # Quality checks
        face_region = gray[y:y+h, x:x+w]
        
        # 1. Size validation
        if w < self.face_size_threshold[0] or h < self.face_size_threshold[1]:
            return False, (x, y, w, h), "Face too small - move closer"
        
        # 2. Position validation (face should be roughly centered)
        frame_center_x = frame.shape[1] // 2
        face_center_x = x + w // 2
        if abs(face_center_x - frame_center_x) > frame.shape[1] * 0.2:
            return False, (x, y, w, h), "Please center your face"
        
        # 3. Blur detection using Laplacian variance
        blur_value = cv2.Laplacian(face_region, cv2.CV_64F).var()
        if blur_value < self.blur_threshold:
            return False, (x, y, w, h), "Image too blurry - hold still"
        
        # 4. Brightness validation
        brightness = np.mean(face_region)
        if brightness < self.brightness_range[0]:
            return False, (x, y, w, h), "Too dark - improve lighting"
        elif brightness > self.brightness_range[1]:
            return False, (x, y, w, h), "Too bright - reduce lighting"
        
        return True, (x, y, w, h), "Good quality - capturing..."
    
    def preprocess_face_image(self, frame, face_coords):
        """
        Step 3: Image Pre-processing
        Extracts and enhances face region for training
        """
        x, y, w, h = face_coords
        
        # 1. Extract face region with padding
        padding = int(0.2 * max(w, h))  # 20% padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # 2. Convert to grayscale and extract face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        student_dir = f"{base_dir}/{student_id}_{student_name}"
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(student_dir):
            os.makedirs(student_dir)
            
        return student_dir
    
    def enroll_student(self, student_name, student_id):
        """
        Complete enrollment process with guided prompts
        Integrates Steps 1, 2, and 3
        """
        print(f"\n=== ENROLLING STUDENT: {student_name} (ID: {student_id}) ===")
        print("Press 'q' to quit, 'n' for next prompt, 's' to skip current prompt")
        
        # Create student directory
        student_dir = self.create_student_directory(student_name, student_id)
        
        # Initialize camera (Step 1: Video Capture)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        total_images_captured = 0
        
        # Go through each enrollment prompt
        for prompt_idx, prompt in enumerate(self.enrollment_prompts):
            print(f"\n--- Prompt {prompt_idx + 1}/{len(self.enrollment_prompts)}: {prompt} ---")
            
            images_for_this_pose = 0
            last_capture_time = 0
            capture_interval = 0.2  # Capture every 0.2 seconds when quality is good
            
            while images_for_this_pose < self.target_images_per_pose:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Step 2: Face Detection and Validation
                face_detected, face_coords, status_message = self.detect_and_validate_face(frame)
                
                # Draw feedback on frame
                if face_detected:
                    x, y, w, h = face_coords
                    # Green rectangle for good quality
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Auto-capture if enough time has passed
                    current_time = time.time()
                    if current_time - last_capture_time > capture_interval:
                        # Step 3: Image Pre-processing
                        processed_face = self.preprocess_face_image(frame, face_coords)
                        
                        # Save processed image
                        filename = f"{student_dir}/pose_{prompt_idx}_{images_for_this_pose:03d}.jpg"
                        cv2.imwrite(filename, processed_face)
                        
                        images_for_this_pose += 1
                        total_images_captured += 1
                        last_capture_time = current_time
                        
                        print(f"Captured: {images_for_this_pose}/{self.target_images_per_pose} for current pose")
                
                elif face_coords:
                    x, y, w, h = face_coords
                    # Red rectangle for poor quality
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Display status and prompt
                cv2.putText(frame, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, status_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if face_detected else (0, 0, 255), 2)
                cv2.putText(frame, f"Images: {images_for_this_pose}/{self.target_images_per_pose}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow('Student Enrollment', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                elif key == ord('n') and images_for_this_pose >= 5:  # Allow skip if at least 5 images
                    break
                elif key == ord('s'):
                    break
            
            print(f"Completed pose: {prompt}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n=== ENROLLMENT COMPLETE ===")
        print(f"Total images captured: {total_images_captured}")
        print(f"Images saved in: {student_dir}")
        
        return True

def main():
    """Demo the enrollment system"""
    enrollment_system = EnrollmentSystem()
    
    print("=== AI ATTENDANCE SYSTEM - ENROLLMENT PHASE ===")
    print("This demonstrates real-time face detection and image pre-processing")
    print()
    
    # Get student details
    student_name = input("Enter student name: ").strip()
    student_id = input("Enter student ID: ").strip()
    
    if not student_name or not student_id:
        print("Error: Please provide both name and ID")
        return
    
    # Start enrollment
    success = enrollment_system.enroll_student(student_name, student_id)
    
    if success:
        print("Enrollment completed successfully!")
        print("You can now test recognition with the captured data.")
    else:
        print("Enrollment was cancelled or failed.")

if __name__ == "__main__":
    main()
