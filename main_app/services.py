"""
Face Detection and Image Processing Services for Django Attendance System
Converted from Streamlit StudentEnrollmentSystem
"""

import cv2
import os
import numpy as np
from datetime import datetime
import tempfile
import base64
from io import BytesIO
from PIL import Image
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import logging

logger = logging.getLogger(__name__)


class FaceDetectionService:
    """Service class for face detection and image processing"""
    
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load configuration dynamically
        self._load_configuration()
        
        # Optimized prompts for faster, comprehensive face capture
        # Reduced from 10 to 7 prompts covering essential angles
        self.enrollment_prompts = [
            "Look straight ahead at the camera",
            "Look slightly up and smile", 
            "Look slightly down with neutral expression",
            "Turn your head slightly to the left",
            "Turn your head slightly to the right",
            "Tilt your head slightly left, then right",
            "Final pose: natural smile at camera"
        ]
    
    def _load_configuration(self):
        """Load configuration from dynamic settings or fallback to Django settings"""
        try:
            from .config_service import config_service
            # Use dynamic configuration service
            enrollment_settings = config_service.get_enrollment_settings()
            performance_settings = config_service.get_performance_settings()
            
            self.target_images_total = enrollment_settings['TARGET_IMAGES_TOTAL']
            self.images_per_prompt = enrollment_settings['IMAGES_PER_PROMPT']
            self.face_size_threshold = enrollment_settings['FACE_SIZE_THRESHOLD']
            self.image_size = enrollment_settings['IMAGE_SIZE']
            self.blur_threshold = enrollment_settings['BLUR_THRESHOLD']
            self.brightness_range = enrollment_settings['BRIGHTNESS_RANGE']
            self.low_light_threshold = enrollment_settings['LOW_LIGHT_THRESHOLD']
            
            # Performance settings
            self.fps_target = performance_settings['CAMERA_FPS']
            self.processing_timeout = performance_settings['PROCESSING_TIMEOUT']
            
            # Derived settings
            self.auto_capture_delay = 0.8
            self.quality_threshold = 85
            
            logger.info("Loaded dynamic configuration successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load dynamic configuration: {str(e)}, using Django settings")
            # Fallback to Django configuration
            enrollment_settings = getattr(settings, 'ENROLLMENT_SETTINGS', {})
            self.target_images_total = enrollment_settings.get('TARGET_IMAGES_TOTAL', 100)
            self.images_per_prompt = enrollment_settings.get('IMAGES_PER_PROMPT', 15)
            self.face_size_threshold = enrollment_settings.get('FACE_SIZE_THRESHOLD', (50, 50))
            self.image_size = enrollment_settings.get('IMAGE_SIZE', (128, 128))
            self.blur_threshold = enrollment_settings.get('BLUR_THRESHOLD', 80)
            self.brightness_range = enrollment_settings.get('BRIGHTNESS_RANGE', (15, 230))
            self.low_light_threshold = enrollment_settings.get('LOW_LIGHT_THRESHOLD', 50)
            self.auto_capture_delay = enrollment_settings.get('AUTO_CAPTURE_DELAY', 0.8)
            self.quality_threshold = enrollment_settings.get('QUALITY_THRESHOLD', 85)
            self.fps_target = 30
            self.processing_timeout = 30
    
    def reload_configuration(self):
        """Reload configuration from dynamic settings"""
        self._load_configuration()
        logger.info("Configuration reloaded")
    
    def enhance_face_image(self, face_region, target_size=None):
        """
        Enhanced preprocessing for face images to improve recognition accuracy
        Implements requirement: Data Pre-processing for improved accuracy
        """
        if target_size is None:
            target_size = self.image_size
            
        try:
            # 1. Noise reduction using bilateral filter
            denoised = cv2.bilateralFilter(face_region, 9, 75, 75)
            
            # 2. Histogram equalization for better contrast
            equalized = cv2.equalizeHist(denoised)
            
            # 3. Gaussian blur for smoothing
            smoothed = cv2.GaussianBlur(equalized, (3, 3), 0)
            
            # 4. Resize to target size with high-quality interpolation
            resized = cv2.resize(smoothed, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # 5. Normalize pixel values for consistency
            normalized = resized.astype(np.float32)
            normalized = cv2.normalize(normalized, normalized, 0, 255, cv2.NORM_MINMAX)
            normalized = normalized.astype(np.uint8)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in face enhancement: {str(e)}")
            # Fallback to simple resize
            return cv2.resize(face_region, target_size, interpolation=cv2.INTER_AREA)
    
    def calculate_face_quality_score(self, face_region, x, y, w, h, frame_shape):
        """
        Calculate comprehensive quality score for face detection
        Implements requirement: Face Detection with quality assessment
        """
        score = 0
        quality_metrics = {}
        
        # 1. Size quality (25 points)
        size_ratio = (w * h) / (frame_shape[0] * frame_shape[1])
        if size_ratio > 0.1:  # Face takes up good portion of frame
            score += 25
        elif size_ratio > 0.05:
            score += 15
        else:
            score += 5
        quality_metrics['size_score'] = size_ratio * 100
        
        # 2. Position quality (25 points)
        frame_center_x, frame_center_y = frame_shape[1] // 2, frame_shape[0] // 2
        face_center_x, face_center_y = x + w // 2, y + h // 2
        center_distance = np.sqrt((face_center_x - frame_center_x)**2 + (face_center_y - frame_center_y)**2)
        max_distance = np.sqrt(frame_shape[1]**2 + frame_shape[0]**2) * 0.3
        
        if center_distance < max_distance * 0.5:
            score += 25
        elif center_distance < max_distance:
            score += 15
        else:
            score += 5
        quality_metrics['position_score'] = max(0, 100 - (center_distance / max_distance * 100))
        
        # 3. Sharpness quality (25 points)
        blur_value = cv2.Laplacian(face_region, cv2.CV_64F).var()
        if blur_value > self.blur_threshold * 1.5:
            score += 25
        elif blur_value > self.blur_threshold:
            score += 15
        else:
            score += 5
        quality_metrics['sharpness_score'] = min(100, blur_value)
        
        # 4. Lighting quality (25 points)
        brightness = np.mean(face_region)
        optimal_range = (60, 180)  # Optimal brightness range
        if optimal_range[0] <= brightness <= optimal_range[1]:
            score += 25
        elif self.brightness_range[0] <= brightness <= self.brightness_range[1]:
            score += 15
        else:
            score += 5
        quality_metrics['lighting_score'] = min(100, brightness / 2)
        
        quality_metrics['total_score'] = score
        return score, quality_metrics
    
    def extract_face_features(self, face_image):
        """
        Extract basic face features for consistency validation
        Implements requirement: Face Recognition - basic feature extraction
        """
        try:
            # Convert to appropriate size for feature extraction
            face_resized = cv2.resize(face_image, (100, 100))
            
            # Calculate basic geometric features
            features = {
                'brightness_mean': float(np.mean(face_resized)),
                'brightness_std': float(np.std(face_resized)),
                'contrast': float(np.std(face_resized) / np.mean(face_resized)) if np.mean(face_resized) > 0 else 0,
                'edge_density': float(np.mean(cv2.Canny(face_resized, 50, 150))),
                'histogram_features': [float(x) for x in cv2.calcHist([face_resized], [0], None, [16], [0, 256]).flatten()]
            }
            
            return features
        except Exception as e:
            logger.error(f"Error extracting face features: {str(e)}")
            return None
    
    def validate_face_consistency(self, student, current_features):
        """
        Validate if current face is consistent with previously enrolled faces
        Implements requirement: Face Recognition - consistency validation
        """
        from .models import EnrollmentImage
        
        try:
            # Get previously enrolled images for this student
            previous_images = EnrollmentImage.objects.filter(student=student).order_by('-captured_at')[:5]
            
            if len(previous_images) < 2:
                return True, "Insufficient data for consistency check"
            
            # For now, implement basic consistency check
            # In production, you would use more sophisticated face recognition
            similarities = []
            
            for prev_image in previous_images:
                # Basic similarity based on brightness and contrast patterns
                if hasattr(prev_image, 'brightness_score'):
                    brightness_diff = abs(current_features.get('brightness_mean', 0) - prev_image.brightness_score)
                    if brightness_diff < 30:  # Reasonable range
                        similarities.append(True)
                    else:
                        similarities.append(False)
            
            consistency_score = sum(similarities) / len(similarities) if similarities else 0
            
            if consistency_score > 0.6:  # 60% similarity threshold
                return True, f"Face consistency validated (Score: {consistency_score:.2f})"
            else:
                return False, f"Face appears inconsistent with previous enrollments (Score: {consistency_score:.2f})"
                
        except Exception as e:
            logger.error(f"Error in face consistency validation: {str(e)}")
            return True, "Consistency check skipped due to error"

    def process_image_from_base64(self, base64_data):
        """
        Process base64 image data and return OpenCV image
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            # Decode base64 to image
            image_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_data))
            
            # Convert to OpenCV format
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:  # RGB
                    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                elif image_array.shape[2] == 4:  # RGBA
                    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                else:
                    raise ValueError("Unsupported image format")
            else:
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            return image_cv
        except Exception as e:
            logger.error(f"Error processing base64 image: {str(e)}")
            return None
    
    def detect_and_validate_face(self, frame):
        """
        Enhanced face detection with quality scoring and preprocessing
        Implements: Face Detection, Data Collection, Data Pre-processing
        Returns: (is_valid, face_coords, message, confidence_score, quality_metrics)
        """
        if frame is None:
            return False, None, "❌ Invalid frame received", 0, {}
        
        try:
            # Step 3: Image Pre-Processing for 30 FPS stream
            # Apply real-time preprocessing optimized for video frames
            preprocessed_frame = self.preprocess_frame_30fps(frame)
            
            # Use preprocessed frame for face detection
            gray = preprocessed_frame
            
            # Enhanced face detection with optimized parameters for 30 FPS
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,  # Smaller scale factor for better detection
                minNeighbors=6,    # Increased for more stable detection
                minSize=self.face_size_threshold,
                flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
            )
            
        except Exception as e:
            logger.error(f"Error in 30 FPS face detection: {str(e)}")
            return False, None, f"❌ Face detection error: {str(e)}", 0, {}
        
        if len(faces) == 0:
            return False, None, "❌ No face detected - please position yourself in front of camera", 0, {}
        elif len(faces) > 1:
            return False, None, "❌ Multiple faces detected - ensure only one person is visible", 0, {}
        
        try:
            # Get the detected face
            (x, y, w, h) = faces[0]
            # Convert NumPy types to native Python types for JSON serialization
            x, y, w, h = int(x), int(y), int(w), int(h)
            face_region = gray[y:y+h, x:x+w]
            
            # Calculate comprehensive quality score
            quality_score, quality_metrics = self.calculate_face_quality_score(
                face_region, x, y, w, h, frame.shape
            )
            
            # Enhanced preprocessing for the detected face
            enhanced_face = self.enhance_face_image(face_region)
            
        except Exception as e:
            logger.error(f"Error in 30 FPS face validation: {str(e)}")
            return False, None, f"❌ Face validation error: {str(e)}", 0, {}
        
        # Quality threshold check
        if quality_score < self.quality_threshold:
            reasons = []
            if quality_metrics.get('size_score', 0) < 30:
                reasons.append("face too small")
            if quality_metrics.get('position_score', 0) < 50:
                reasons.append("not centered")
            if quality_metrics.get('sharpness_score', 0) < 50:
                reasons.append("too blurry")
            if quality_metrics.get('lighting_score', 0) < 40:
                reasons.append("poor lighting")
            
            # Ensure quality metrics are JSON serializable for failure case too
            quality_metrics_safe = {}
            for key, value in quality_metrics.items():
                if hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                    quality_metrics_safe[key] = int(value)
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                    quality_metrics_safe[key] = float(value)
                elif isinstance(value, np.ndarray):
                    quality_metrics_safe[key] = value.tolist()
                elif isinstance(value, (int, float)):
                    quality_metrics_safe[key] = value
                else:
                    try:
                        quality_metrics_safe[key] = float(value)
                    except:
                        quality_metrics_safe[key] = value
            
            message = f"❌ Quality too low: {', '.join(reasons)} (Score: {quality_score}/100)"
            return False, (x, y, w, h), message, int(quality_score), quality_metrics_safe
        
        # Success - high quality face detected
        confidence_score = int(quality_score)
        
        # Ensure all quality metrics are JSON serializable
        quality_metrics_safe = {}
        for key, value in quality_metrics.items():
            if hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                quality_metrics_safe[key] = int(value)
            elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                quality_metrics_safe[key] = float(value)
            elif isinstance(value, np.ndarray):
                quality_metrics_safe[key] = value.tolist()
            elif isinstance(value, (int, float)):
                quality_metrics_safe[key] = value
            else:
                # Try to convert to float, fallback to original value
                try:
                    quality_metrics_safe[key] = float(value)
                except:
                    quality_metrics_safe[key] = value
        
        # Add additional metrics
        quality_metrics_safe['blur_value'] = float(cv2.Laplacian(face_region, cv2.CV_64F).var())
        quality_metrics_safe['brightness_score'] = float(np.mean(face_region))
        quality_metrics_safe['confidence_score'] = int(confidence_score)
        
        return True, (x, y, w, h), f"✅ High quality face detected (Score: {quality_score}/100)", confidence_score, quality_metrics_safe
    
    def preprocess_frame_30fps(self, frame):
        """
        Step 3: Image Pre-Processing for 30 FPS video stream
        Optimized for real-time processing of video frames with error handling
        """
        try:
            # Convert to grayscale immediately for faster processing
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
                
            # Apply basic enhancement for better face detection
            # 1. Histogram equalization for contrast improvement
            enhanced_frame = cv2.equalizeHist(gray_frame)
            
            # 2. Gaussian blur to reduce noise (light filtering for speed)
            enhanced_frame = cv2.GaussianBlur(enhanced_frame, (3, 3), 0)
            
            # 3. Check lighting conditions and apply adaptive enhancement
            brightness = np.mean(enhanced_frame)
            
            if brightness < 80:  # Low light conditions
                # Apply CLAHE for low-light enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                enhanced_frame = clahe.apply(enhanced_frame)
            elif brightness > 200:  # High light conditions
                # Reduce brightness slightly
                enhanced_frame = np.clip(enhanced_frame * 0.9, 0, 255).astype(np.uint8)
                
            return enhanced_frame
            
        except Exception as e:
            logger.error(f"Error in 30 FPS preprocessing: {str(e)}")
            # Return original frame as grayscale if preprocessing fails
            if len(frame.shape) == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame

    def preprocess_face_image(self, frame, face_coords):
        """
        Advanced image preprocessing with low-light enhancement
        Returns: processed face image and quality metrics
        """
        x, y, w, h = face_coords
        
        # Extract face region with padding
        padding = int(0.25 * max(w, h))  # 25% padding for better context
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Step 3: Image Pre-Processing - Convert to grayscale as required
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
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
    
    def save_processed_image(self, processed_image, student, prompt_index, image_sequence, 
                           face_coords, quality_metrics, capture_mode='live'):
        """
        Save processed face image to Django model
        Returns: EnrollmentImage instance
        """
        from .models import EnrollmentImage
        
        # Convert processed image to bytes
        success, encoded_image = cv2.imencode('.jpg', processed_image)
        if not success:
            raise ValueError("Failed to encode processed image")
        
        image_bytes = encoded_image.tobytes()
        
        # Create ContentFile
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"face_{student.student_id}_{prompt_index:02d}_{image_sequence:03d}_{timestamp}.jpg"
        image_file = ContentFile(image_bytes, name=filename)
        
        # Create EnrollmentImage instance
        enrollment_image = EnrollmentImage(
            student=student,
            prompt_index=prompt_index,
            image_sequence=image_sequence,
            confidence_score=quality_metrics.get('confidence_score', 0),
            face_coordinates=face_coords,
            brightness_score=quality_metrics.get('brightness_score', 0),
            blur_score=quality_metrics.get('blur_score', 0),
            capture_mode=capture_mode
        )
        
        # Save the image file
        enrollment_image.image.save(filename, image_file, save=False)
        enrollment_image.save()
        
        return enrollment_image
    
    def get_enrollment_progress(self, student):
        """
        Get enrollment progress for a student
        Returns: dict with progress information
        """
        from .models import EnrollmentImage
        
        total_images = EnrollmentImage.objects.filter(student=student).count()
        
        # Get progress per prompt
        prompt_progress = {}
        for i, prompt in enumerate(self.enrollment_prompts):
            prompt_images = EnrollmentImage.objects.filter(
                student=student, 
                prompt_index=i
            ).count()
            prompt_progress[i] = {
                'prompt_text': prompt,
                'images_captured': int(prompt_images),
                'target_images': int(self.images_per_prompt),
                'completed': bool(prompt_images >= self.images_per_prompt),
                'progress_percentage': float(min(100, (prompt_images / self.images_per_prompt) * 100))
            }
        
        # Calculate overall progress
        progress_percentage = float((total_images / self.target_images_total) * 100)
        enrollment_completed = bool(total_images >= self.target_images_total)
        
        # Current prompt (first incomplete prompt)
        current_prompt_index = 0
        for i, prompt_data in prompt_progress.items():
            if not prompt_data['completed']:
                current_prompt_index = i
                break
        else:
            current_prompt_index = len(self.enrollment_prompts) - 1
        
        return {
            'total_images': int(total_images),
            'target_images': int(self.target_images_total),
            'progress_percentage': float(progress_percentage),
            'enrollment_completed': bool(enrollment_completed),
            'current_prompt_index': int(current_prompt_index),
            'current_prompt_text': str(self.enrollment_prompts[current_prompt_index] if current_prompt_index < len(self.enrollment_prompts) else "Complete"),
            'prompt_progress': prompt_progress,
            'prompts_completed': int(sum(1 for p in prompt_progress.values() if p['completed'])),
            'total_prompts': int(len(self.enrollment_prompts))
        }
    
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
    
    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string for web display"""
        if frame is None:
            return None
        
        try:
            # Convert BGR to RGB for web display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.error(f"Error converting frame to base64: {str(e)}")
            return None


# Service instance
face_detection_service = FaceDetectionService()
