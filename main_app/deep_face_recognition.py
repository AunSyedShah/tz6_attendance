"""
Deep Learning Face Recognition Service for AI Attendance System
Uses pre-trained models through DeepFace library for high accuracy recognition
"""
import os
import cv2
import numpy as np
import pickle
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import time

# Core dependencies
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Use CPU for stability
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

from django.conf import settings

logger = logging.getLogger(__name__)

class DeepFaceRecognitionService:
    """
    Advanced Deep Learning Face Recognition Service
    Uses DeepFace with pre-trained models for high accuracy recognition
    """
    
    def __init__(self):
        self.model_path = os.path.join(settings.MEDIA_ROOT, 'deep_models')
        self.embeddings_db = os.path.join(self.model_path, 'face_embeddings.db')
        self.config_file = os.path.join(self.model_path, 'model_config.json')
        
        # Model configurations - available pre-trained models
        self.available_models = {
            'Facenet': {'threshold': 0.40, 'size': 160, 'dims': 128},
            'Facenet512': {'threshold': 0.30, 'size': 160, 'dims': 512},
            'ArcFace': {'threshold': 0.68, 'size': 112, 'dims': 512},
            'VGG-Face': {'threshold': 0.60, 'size': 224, 'dims': 4096},
            'OpenFace': {'threshold': 0.10, 'size': 96, 'dims': 128},
            'DeepFace': {'threshold': 0.23, 'size': 152, 'dims': 4096}
        }
        
        # Default configuration - Facenet512 for best balance
        self.model_name = 'Facenet512'
        self.detector_backend = 'opencv'  # opencv, mtcnn, retinaface
        self.distance_metric = 'cosine'
        self.threshold = self.available_models[self.model_name]['threshold']
        
        # Recognition settings
        self.confidence_threshold = 0.7     # 70% minimum for recognition
        self.attendance_threshold = 0.85    # 85% for auto-attendance
        
        # Initialize system
        if not DEEPFACE_AVAILABLE:
            logger.warning("DeepFace not available. Deep learning features disabled.")
            return
            
        os.makedirs(self.model_path, exist_ok=True)
        self.init_database()
        self.load_configuration()
        
        logger.info(f"DeepFace recognition service initialized with {self.model_name}")
    
    def get_model_info(self):
        """Get current model information"""
        if not DEEPFACE_AVAILABLE:
            return {'available': False, 'error': 'DeepFace not installed'}
            
        return {
            'available': True,
            'model_name': self.model_name,
            'detector': self.detector_backend,
            'threshold': self.threshold,
            'distance_metric': self.distance_metric,
            'available_models': list(self.available_models.keys())
        }
    
    def init_database(self):
        """Initialize SQLite database for storing face embeddings"""
        try:
            conn = sqlite3.connect(self.embeddings_db)
            cursor = conn.cursor()
            
            # Create embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    image_path TEXT,
                    quality_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(student_id, image_path)
                )
            ''')
            
            # Create students table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students_deep (
                    student_id TEXT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    enrollment_count INTEGER DEFAULT 0,
                    last_training TIMESTAMP,
                    model_accuracy REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Deep learning database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def load_configuration(self):
        """Load model configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.model_name = config.get('model_name', self.model_name)
                    self.detector_backend = config.get('detector_backend', self.detector_backend)
                    self.threshold = config.get('threshold', self.threshold)
                    logger.info(f"Loaded configuration: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def save_configuration(self):
        """Save current configuration to file"""
        try:
            config = {
                'model_name': self.model_name,
                'detector_backend': self.detector_backend,
                'threshold': self.threshold,
                'distance_metric': self.distance_metric,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get_model_statistics(self) -> Dict:
        """Get statistics about the trained model"""
        try:
            if not os.path.exists(self.embeddings_db):
                return {
                    'model_name': self.model_name,
                    'total_students': 0,
                    'total_embeddings': 0,
                    'threshold': self.threshold,
                    'students_detail': [],
                    'average_embeddings_per_student': 0
                }
                
            conn = sqlite3.connect(self.embeddings_db)
            cursor = conn.cursor()
            
            # Get student count
            cursor.execute('SELECT COUNT(DISTINCT student_id) FROM face_embeddings')
            student_count = cursor.fetchone()[0]
            
            # Get total embeddings
            cursor.execute('SELECT COUNT(*) FROM face_embeddings WHERE model_name = ?', (self.model_name,))
            embedding_count = cursor.fetchone()[0]
            
            # Get model info
            cursor.execute('''
                SELECT student_id, student_name, COUNT(*) as embedding_count
                FROM face_embeddings 
                WHERE model_name = ?
                GROUP BY student_id, student_name
            ''', (self.model_name,))
            
            students_data = cursor.fetchall()
            conn.close()
            
            return {
                'model_name': self.model_name,
                'total_students': student_count,
                'total_embeddings': embedding_count,
                'threshold': self.threshold,
                'students_detail': [
                    {'student_id': row[0], 'name': row[1], 'embeddings': row[2]}
                    for row in students_data
                ],
                'average_embeddings_per_student': round(embedding_count / max(student_count, 1), 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting model statistics: {str(e)}")
            return {'error': str(e)}
    
    def train_deep_model(self) -> bool:
        """
        Train the deep learning model with all enrolled students
        
        Returns:
            True if training successful, False otherwise
        """
        if not DEEPFACE_AVAILABLE:
            logger.error("DeepFace not available for training")
            return False
            
        try:
            from .models import Student
            
            students = Student.objects.filter(enrollment_completed=True)
            
            if not students.exists():
                logger.warning("No enrolled students found for training")
                return False
            
            total_embeddings = 0
            
            for student in students:
                embeddings_count = self.generate_embeddings_for_student(student.student_id)
                total_embeddings += embeddings_count
                
                logger.info(f"Processed student {student.student_id}: {embeddings_count} embeddings")
            
            # Save configuration
            self.save_configuration()
            
            logger.info(f"Deep learning model training completed. Total embeddings: {total_embeddings}")
            return total_embeddings > 0
            
        except Exception as e:
            logger.error(f"Error training deep model: {str(e)}")
            return False
    
    def generate_embeddings_for_student(self, student_id: str) -> int:
        """
        Generate face embeddings for a specific student from their enrollment images
        
        Args:
            student_id: Student ID to process
            
        Returns:
            Number of embeddings generated
        """
        if not DEEPFACE_AVAILABLE:
            logger.error("DeepFace not available")
            return 0
            
        try:
            from .models import Student, EnrollmentImage
            
            # Get student and their images
            student = Student.objects.get(student_id=student_id)
            images = EnrollmentImage.objects.filter(student=student)
            
            if not images.exists():
                logger.warning(f"No enrollment images found for student {student_id}")
                return 0
            
            embeddings_generated = 0
            conn = sqlite3.connect(self.embeddings_db)
            cursor = conn.cursor()
            
            for image in images:
                try:
                    image_path = image.image.path
                    
                    # Skip if embedding already exists
                    cursor.execute('''
                        SELECT id FROM face_embeddings 
                        WHERE student_id = ? AND image_path = ?
                    ''', (student_id, image_path))
                    
                    if cursor.fetchone():
                        continue  # Skip existing embeddings
                    
                    # Extract embedding
                    embedding = self.extract_face_embedding(image_path)
                    
                    if embedding is not None:
                        # Store embedding in database
                        embedding_blob = pickle.dumps(embedding)
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO face_embeddings 
                            (student_id, student_name, model_name, embedding, image_path, quality_score)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            student_id,
                            student.full_name,
                            self.model_name,
                            embedding_blob,
                            image_path,
                            image.confidence_score
                        ))
                        
                        embeddings_generated += 1
                        
                except Exception as e:
                    logger.error(f"Error processing image {image.image.path}: {str(e)}")
                    continue
            
            # Update student record
            cursor.execute('''
                INSERT OR REPLACE INTO students_deep 
                (student_id, student_name, enrollment_count, last_training)
                VALUES (?, ?, ?, ?)
            ''', (student_id, student.full_name, embeddings_generated, datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Generated {embeddings_generated} embeddings for student {student_id}")
            return embeddings_generated
            
        except Exception as e:
            logger.error(f"Error generating embeddings for student {student_id}: {str(e)}")
            return 0
    
    def extract_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face embedding from image using DeepFace
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Face embedding as numpy array or None if failed
        """
        if not DEEPFACE_AVAILABLE:
            logger.error("DeepFace not available")
            return None
            
        try:
            # Use DeepFace to extract embedding
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                # Return the first (and typically only) embedding
                embedding = embedding_objs[0]['embedding']
                return np.array(embedding)
            else:
                logger.warning(f"No face found in {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting embedding from {image_path}: {str(e)}")
            return None


# Global service instance
deep_face_service = DeepFaceRecognitionService()
