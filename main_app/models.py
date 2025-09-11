from django.db import models
from django.core.validators import RegexValidator
from django.utils import timezone
from django.conf import settings
import os
import shutil
import logging

logger = logging.getLogger(__name__)


class Student(models.Model):
    """Model to store student information"""
    student_id = models.CharField(
        max_length=20, 
        unique=True,
        validators=[
            RegexValidator(
                regex=r'^[A-Z0-9]+$',
                message='Student ID must contain only uppercase letters and numbers'
            )
        ]
    )
    full_name = models.CharField(max_length=100)
    email = models.EmailField(blank=True, null=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    enrollment_date = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    
    # Enrollment status
    enrollment_completed = models.BooleanField(default=False)
    total_images_captured = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-enrollment_date']
        verbose_name = 'Student'
        verbose_name_plural = 'Students'
    
    def __str__(self):
        return f"{self.student_id} - {self.full_name}"
    
    def get_enrollment_directory(self):
        """Get the directory path for storing this student's images"""
        clean_name = "".join(c for c in self.full_name if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_name = clean_name.replace(' ', '_')
        return f"enrollment_data/{self.student_id}_{clean_name}"
    
    def delete(self, *args, **kwargs):
        """Delete student and all associated files/directories"""
        try:
            # 1. Delete all enrollment images (this will trigger their individual delete methods)
            enrollment_images = self.enrollment_images.all()
            image_count = enrollment_images.count()
            
            logger.info(f"Deleting student {self.student_id} with {image_count} images")
            
            # Delete each enrollment image (triggers file deletion)
            for image in enrollment_images:
                try:
                    image.delete()
                except Exception as e:
                    logger.error(f"Error deleting image {image.id}: {str(e)}")
            
            # 2. Delete enrollment directory if it exists
            enrollment_dir = os.path.join(settings.MEDIA_ROOT, self.get_enrollment_directory())
            if os.path.exists(enrollment_dir):
                try:
                    shutil.rmtree(enrollment_dir)
                    logger.info(f"Deleted enrollment directory: {enrollment_dir}")
                except Exception as e:
                    logger.error(f"Error deleting directory {enrollment_dir}: {str(e)}")
            
            # 3. Delete from deep learning database if exists
            try:
                from .deep_face_recognition import deep_face_service
                if hasattr(deep_face_service, 'embeddings_db'):
                    import sqlite3
                    conn = sqlite3.connect(deep_face_service.embeddings_db)
                    cursor = conn.cursor()
                    
                    # Delete embeddings for this student
                    cursor.execute('DELETE FROM face_embeddings WHERE student_id = ?', (self.student_id,))
                    cursor.execute('DELETE FROM students_deep WHERE student_id = ?', (self.student_id,))
                    
                    deleted_embeddings = cursor.rowcount
                    conn.commit()
                    conn.close()
                    
                    if deleted_embeddings > 0:
                        logger.info(f"Deleted {deleted_embeddings} face embeddings for student {self.student_id}")
                        
            except Exception as e:
                logger.error(f"Error deleting deep learning data for student {self.student_id}: {str(e)}")
            
            # 4. Clear any cached recognition data
            try:
                from .config_service import config_service
                config_service.clear_cache()
            except Exception as e:
                logger.warning(f"Could not clear config cache: {str(e)}")
            
            logger.info(f"Successfully deleted student {self.student_id} and all associated data")
            
        except Exception as e:
            logger.error(f"Error during student deletion {self.student_id}: {str(e)}")
            # Continue with deletion even if cleanup fails
        
        # Finally, delete the student record
        super().delete(*args, **kwargs)


class EnrollmentImage(models.Model):
    """Model to store enrollment face images"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='enrollment_images')
    image = models.ImageField(upload_to='enrollment_images/')
    
    # Image metadata
    prompt_index = models.IntegerField()  # Which prompt this image was captured for
    image_sequence = models.IntegerField()  # Sequence number within the prompt
    confidence_score = models.FloatField()  # Face detection confidence
    
    # Quality metrics
    face_coordinates = models.JSONField()  # Store x, y, w, h of detected face
    brightness_score = models.FloatField()
    blur_score = models.FloatField()
    
    # Capture metadata
    capture_timestamp = models.DateTimeField(default=timezone.now)
    capture_mode = models.CharField(
        max_length=10,
        choices=[('live', 'Live Camera'), ('demo', 'Demo Upload')],
        default='live'
    )
    
    class Meta:
        ordering = ['student', 'prompt_index', 'capture_timestamp']
        # Removed unique_together constraint to allow 30 FPS rapid capture
        # Instead, we'll use timestamp-based uniqueness
    
    def __str__(self):
        return f"{self.student.student_id} - Prompt {self.prompt_index} - Image {self.image_sequence} - {self.capture_timestamp}"
    
    def delete(self, *args, **kwargs):
        """Delete the image file when the model instance is deleted"""
        try:
            if self.image:
                # Get the full file path
                file_path = self.image.path
                
                # Delete the physical file
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted image file: {file_path}")
                
                # Also try to delete from default storage (in case of cloud storage)
                try:
                    from django.core.files.storage import default_storage
                    if default_storage.exists(self.image.name):
                        default_storage.delete(self.image.name)
                except Exception as e:
                    logger.warning(f"Could not delete from storage: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error deleting image file for EnrollmentImage {self.id}: {str(e)}")
        
        # Delete the database record
        super().delete(*args, **kwargs)


class EnrollmentSession(models.Model):
    """Model to track enrollment sessions"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='enrollment_sessions')
    session_start = models.DateTimeField(default=timezone.now)
    session_end = models.DateTimeField(null=True, blank=True)
    session_mode = models.CharField(
        max_length=10,
        choices=[('live', 'Live Camera'), ('demo', 'Demo Upload')],
        default='live'
    )
    
    # Session statistics
    total_attempts = models.IntegerField(default=0)
    successful_captures = models.IntegerField(default=0)
    failed_captures = models.IntegerField(default=0)
    session_completed = models.BooleanField(default=False)
    
    # Technical details
    user_agent = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        ordering = ['-session_start']
    
    def __str__(self):
        return f"{self.student.student_id} - {self.session_start.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def success_rate(self):
        """Calculate success rate of captures in this session"""
        if self.total_attempts == 0:
            return 0
        return (self.successful_captures / self.total_attempts) * 100
    
    @property
    def duration(self):
        """Calculate session duration"""
        if self.session_end:
            return self.session_end - self.session_start
        return timezone.now() - self.session_start


class AttendanceRecord(models.Model):
    """Model to track student attendance records"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='attendance_records')
    date = models.DateField(default=timezone.now)
    entry_time = models.TimeField(null=True, blank=True)
    exit_time = models.TimeField(null=True, blank=True)
    
    # Recognition details
    recognition_confidence = models.FloatField(default=0.0)
    recognition_method = models.CharField(
        max_length=20,
        choices=[
            ('face_recognition', 'Face Recognition'),
            ('manual', 'Manual Entry'),
            ('card', 'ID Card'),
        ],
        default='face_recognition'
    )
    
    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=[
            ('present', 'Present'),
            ('absent', 'Absent'),
            ('late', 'Late'),
            ('excused', 'Excused'),
        ],
        default='absent'
    )
    
    # Additional metadata
    total_duration = models.DurationField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date', '-entry_time']
        unique_together = ['student', 'date']  # One record per student per day
    
    def __str__(self):
        return f"{self.student.student_id} - {self.date} - {self.status}"
    
    @property
    def duration_in_minutes(self):
        """Calculate attendance duration in minutes"""
        if self.entry_time and self.exit_time:
            from datetime import datetime, timedelta
            entry_dt = datetime.combine(self.date, self.entry_time)
            exit_dt = datetime.combine(self.date, self.exit_time)
            
            # Handle case where exit is next day
            if exit_dt < entry_dt:
                exit_dt += timedelta(days=1)
            
            duration = exit_dt - entry_dt
            return duration.total_seconds() / 60
        return None
    
    def mark_entry(self, confidence: float = 1.0):
        """Mark student entry with current time"""
        from datetime import datetime
        self.entry_time = datetime.now().time()
        self.status = 'present'
        self.recognition_confidence = confidence
        self.save()
    
    def mark_exit(self):
        """Mark student exit with current time"""
        from datetime import datetime
        self.exit_time = datetime.now().time()
        if self.entry_time:
            # Calculate total duration
            entry_dt = datetime.combine(self.date, self.entry_time)
            exit_dt = datetime.combine(self.date, self.exit_time)
            if exit_dt >= entry_dt:
                self.total_duration = exit_dt - entry_dt
        self.save()


class FaceEncoding(models.Model):
    """Model to store face encodings for students"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='face_encodings')
    enrollment_image = models.ForeignKey(EnrollmentImage, on_delete=models.CASCADE, related_name='face_encodings')
    
    # Face encoding data (128-dimensional vector)
    encoding_vector = models.JSONField()  # Store as JSON array
    encoding_quality = models.FloatField(default=0.0)  # Quality score of the encoding
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['student', 'enrollment_image']  # One encoding per image
    
    def __str__(self):
        return f"{self.student.student_id} - Encoding (Quality: {self.encoding_quality:.2f})"


class AttendanceSession(models.Model):
    """Model to track real-time attendance monitoring sessions"""
    class_name = models.CharField(max_length=100)
    session_date = models.DateField(default=timezone.now)
    start_time = models.TimeField()
    end_time = models.TimeField(null=True, blank=True)
    
    # Session configuration
    recognition_threshold = models.FloatField(default=0.6)  # Minimum confidence for recognition
    auto_mark_attendance = models.BooleanField(default=True)
    
    # Session statistics
    total_recognitions = models.IntegerField(default=0)
    unique_students_detected = models.IntegerField(default=0)
    false_positives = models.IntegerField(default=0)
    
    # Session status
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-session_date', '-start_time']
    
    def __str__(self):
        return f"{self.class_name} - {self.session_date} {self.start_time}"
    
    @property
    def duration(self):
        """Calculate session duration"""
        if self.end_time:
            from datetime import datetime, timedelta
            start_dt = datetime.combine(self.session_date, self.start_time)
            end_dt = datetime.combine(self.session_date, self.end_time)
            
            if end_dt < start_dt:  # Handle next day scenario
                end_dt += timedelta(days=1)
            
            return end_dt - start_dt
        return None


class RecognitionLog(models.Model):
    """Model to log all face recognition attempts for debugging and analysis"""
    attendance_session = models.ForeignKey(AttendanceSession, on_delete=models.CASCADE, related_name='recognition_logs')
    student = models.ForeignKey(Student, on_delete=models.CASCADE, null=True, blank=True)  # Null if unrecognized
    
    # Recognition details
    recognition_time = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField()
    recognition_successful = models.BooleanField()
    
    # Image metadata (optional)
    image_quality_score = models.FloatField(null=True, blank=True)
    face_size = models.CharField(max_length=20, blank=True)  # e.g., "150x150"
    lighting_conditions = models.CharField(max_length=50, blank=True)
    
    # Action taken
    attendance_marked = models.BooleanField(default=False)
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-recognition_time']
    
    def __str__(self):
        student_name = self.student.full_name if self.student else "Unknown"
        return f"{student_name} - {self.recognition_time.strftime('%H:%M:%S')} - {self.confidence_score:.2f}"


class SystemConfiguration(models.Model):
    """Model to store dynamic system configuration settings"""
    
    SETTING_TYPES = [
        ('INTEGER', 'Integer'),
        ('FLOAT', 'Float'),
        ('STRING', 'String'),
        ('BOOLEAN', 'Boolean'),
        ('JSON', 'JSON'),
    ]
    
    # Configuration identity
    key = models.CharField(max_length=100, unique=True, help_text="Unique setting key")
    name = models.CharField(max_length=200, help_text="Human-readable setting name")
    description = models.TextField(help_text="Description of what this setting controls")
    category = models.CharField(max_length=50, default='General', help_text="Setting category for grouping")
    
    # Value storage
    setting_type = models.CharField(max_length=10, choices=SETTING_TYPES, default='STRING')
    value = models.TextField(help_text="Setting value (stored as text, converted by type)")
    default_value = models.TextField(help_text="Default value for this setting")
    
    # Validation
    min_value = models.FloatField(null=True, blank=True, help_text="Minimum value (for numeric types)")
    max_value = models.FloatField(null=True, blank=True, help_text="Maximum value (for numeric types)")
    validation_regex = models.CharField(max_length=500, blank=True, help_text="Regex for value validation")
    
    # Metadata
    is_active = models.BooleanField(default=True)
    requires_restart = models.BooleanField(default=False, help_text="Does changing this setting require a restart?")
    last_modified = models.DateTimeField(auto_now=True)
    last_modified_by = models.CharField(max_length=100, blank=True)
    
    class Meta:
        ordering = ['category', 'name']
        verbose_name = 'System Configuration'
        verbose_name_plural = 'System Configuration'
    
    def __str__(self):
        return f"{self.category} - {self.name}"
    
    def get_value(self):
        """Convert stored value to appropriate Python type"""
        if not self.value:
            return self.get_default_value()
            
        try:
            if self.setting_type == 'INTEGER':
                return int(self.value)
            elif self.setting_type == 'FLOAT':
                return float(self.value)
            elif self.setting_type == 'BOOLEAN':
                return self.value.lower() in ('true', '1', 'yes', 'on')
            elif self.setting_type == 'JSON':
                import json
                return json.loads(self.value)
            else:  # STRING
                return self.value
        except (ValueError, TypeError, json.JSONDecodeError):
            return self.get_default_value()
    
    def get_default_value(self):
        """Convert default value to appropriate Python type"""
        try:
            if self.setting_type == 'INTEGER':
                return int(self.default_value)
            elif self.setting_type == 'FLOAT':
                return float(self.default_value)
            elif self.setting_type == 'BOOLEAN':
                return self.default_value.lower() in ('true', '1', 'yes', 'on')
            elif self.setting_type == 'JSON':
                import json
                return json.loads(self.default_value)
            else:  # STRING
                return self.default_value
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
    
    def set_value(self, new_value, modified_by=None):
        """Set new value with validation"""
        # Convert to string for storage
        if self.setting_type == 'JSON':
            import json
            self.value = json.dumps(new_value)
        else:
            self.value = str(new_value)
            
        if modified_by:
            self.last_modified_by = modified_by
            
        self.save()
    
    @classmethod
    def get_setting(cls, key, default=None):
        """Get a setting value by key"""
        try:
            setting = cls.objects.get(key=key, is_active=True)
            return setting.get_value()
        except cls.DoesNotExist:
            return default
    
    @classmethod
    def set_setting(cls, key, value, modified_by=None):
        """Set a setting value by key"""
        try:
            setting = cls.objects.get(key=key)
            setting.set_value(value, modified_by)
            return True
        except cls.DoesNotExist:
            return False
