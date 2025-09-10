from django.db import models
from django.core.validators import RegexValidator
from django.utils import timezone
import os


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
        if self.image:
            if os.path.isfile(self.image.path):
                os.remove(self.image.path)
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
