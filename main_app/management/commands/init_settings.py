"""
Management command to initialize system configuration settings
"""
from django.core.management.base import BaseCommand
from main_app.models import SystemConfiguration
import json


class Command(BaseCommand):
    help = 'Initialize system configuration with default settings'

    def handle(self, *args, **options):
        """Initialize all system configuration settings"""
        
        settings_to_create = [
            # File Upload Settings
            {
                'key': 'FILE_UPLOAD_MAX_MEMORY_SIZE',
                'name': 'Maximum File Upload Size (MB)',
                'description': 'Maximum file size allowed for uploads in megabytes',
                'category': 'File Upload',
                'setting_type': 'INTEGER',
                'value': '10',
                'default_value': '10',
                'min_value': 1,
                'max_value': 100,
            },
            {
                'key': 'DATA_UPLOAD_MAX_MEMORY_SIZE',
                'name': 'Maximum Data Upload Size (MB)',
                'description': 'Maximum data size allowed for form uploads in megabytes',
                'category': 'File Upload',
                'setting_type': 'INTEGER',
                'value': '10',
                'default_value': '10',
                'min_value': 1,
                'max_value': 100,
            },
            
            # Enrollment Settings
            {
                'key': 'TARGET_IMAGES_TOTAL',
                'name': 'Total Target Images',
                'description': 'Total number of images to capture during enrollment',
                'category': 'Enrollment',
                'setting_type': 'INTEGER',
                'value': '100',
                'default_value': '100',
                'min_value': 10,
                'max_value': 500,
            },
            {
                'key': 'IMAGES_PER_PROMPT',
                'name': 'Images Per Prompt',
                'description': 'Number of images to capture per facial prompt',
                'category': 'Enrollment',
                'setting_type': 'INTEGER',
                'value': '15',
                'default_value': '15',
                'min_value': 5,
                'max_value': 50,
            },
            {
                'key': 'FACE_SIZE_THRESHOLD',
                'name': 'Face Size Threshold',
                'description': 'Minimum face size (width, height) in pixels for valid detection',
                'category': 'Enrollment',
                'setting_type': 'JSON',
                'value': '[50, 50]',
                'default_value': '[50, 50]',
            },
            {
                'key': 'IMAGE_SIZE',
                'name': 'Processed Image Size',
                'description': 'Size to resize images for processing (width, height)',
                'category': 'Enrollment',
                'setting_type': 'JSON',
                'value': '[128, 128]',
                'default_value': '[128, 128]',
            },
            {
                'key': 'BLUR_THRESHOLD',
                'name': 'Blur Detection Threshold',
                'description': 'Threshold for detecting blurry images (higher = more strict)',
                'category': 'Image Quality',
                'setting_type': 'INTEGER',
                'value': '80',
                'default_value': '80',
                'min_value': 10,
                'max_value': 200,
            },
            {
                'key': 'BRIGHTNESS_RANGE_MIN',
                'name': 'Minimum Brightness',
                'description': 'Minimum acceptable brightness value for images',
                'category': 'Image Quality',
                'setting_type': 'INTEGER',
                'value': '15',
                'default_value': '15',
                'min_value': 0,
                'max_value': 255,
            },
            {
                'key': 'BRIGHTNESS_RANGE_MAX',
                'name': 'Maximum Brightness',
                'description': 'Maximum acceptable brightness value for images',
                'category': 'Image Quality',
                'setting_type': 'INTEGER',
                'value': '230',
                'default_value': '230',
                'min_value': 0,
                'max_value': 255,
            },
            {
                'key': 'LOW_LIGHT_THRESHOLD',
                'name': 'Low Light Threshold',
                'description': 'Threshold below which lighting is considered too low',
                'category': 'Image Quality',
                'setting_type': 'INTEGER',
                'value': '50',
                'default_value': '50',
                'min_value': 0,
                'max_value': 255,
            },
            
            # Face Recognition Settings
            {
                'key': 'RECOGNITION_CONFIDENCE_THRESHOLD',
                'name': 'Recognition Confidence Threshold',
                'description': 'Minimum confidence required for face recognition (0.0 - 1.0)',
                'category': 'Face Recognition',
                'setting_type': 'FLOAT',
                'value': '0.7',
                'default_value': '0.7',
                'min_value': 0.1,
                'max_value': 1.0,
            },
            {
                'key': 'ATTENDANCE_AUTO_MARK_THRESHOLD',
                'name': 'Auto-Mark Attendance Threshold',
                'description': 'Confidence threshold for automatically marking attendance',
                'category': 'Face Recognition',
                'setting_type': 'FLOAT',
                'value': '0.85',
                'default_value': '0.85',
                'min_value': 0.5,
                'max_value': 1.0,
            },
            {
                'key': 'DEEP_LEARNING_MODEL',
                'name': 'Deep Learning Model',
                'description': 'Pre-trained model to use for face recognition',
                'category': 'Face Recognition',
                'setting_type': 'STRING',
                'value': 'Facenet512',
                'default_value': 'Facenet512',
                'validation_regex': '^(Facenet|Facenet512|ArcFace|VGG-Face|OpenFace|DeepFace)$',
            },
            
            # Performance Settings
            {
                'key': 'CAMERA_FPS',
                'name': 'Camera Frame Rate',
                'description': 'Target frames per second for camera capture',
                'category': 'Performance',
                'setting_type': 'INTEGER',
                'value': '30',
                'default_value': '30',
                'min_value': 10,
                'max_value': 60,
            },
            {
                'key': 'PROCESSING_TIMEOUT',
                'name': 'Processing Timeout (seconds)',
                'description': 'Maximum time to wait for image processing operations',
                'category': 'Performance',
                'setting_type': 'INTEGER',
                'value': '30',
                'default_value': '30',
                'min_value': 5,
                'max_value': 120,
            },
        ]
        
        created_count = 0
        updated_count = 0
        
        for setting_data in settings_to_create:
            setting, created = SystemConfiguration.objects.get_or_create(
                key=setting_data['key'],
                defaults=setting_data
            )
            
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created setting: {setting_data["name"]}')
                )
            else:
                # Update description and category if changed
                if (setting.description != setting_data['description'] or 
                    setting.category != setting_data['category']):
                    setting.description = setting_data['description']
                    setting.category = setting_data['category']
                    setting.save()
                    updated_count += 1
                    self.stdout.write(
                        self.style.WARNING(f'Updated setting: {setting_data["name"]}')
                    )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'\nInitialization complete!'
                f'\nCreated: {created_count} settings'
                f'\nUpdated: {updated_count} settings'
            )
        )
