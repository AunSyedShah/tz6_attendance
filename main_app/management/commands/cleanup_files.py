"""
Management command to clean up orphaned files and empty directories
"""
from django.core.management.base import BaseCommand
from django.conf import settings
from main_app.signals import cleanup_orphaned_files, cleanup_empty_directories
from main_app.models import Student, EnrollmentImage
import os
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Clean up orphaned files and empty directories in media folder'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting',
        )
        parser.add_argument(
            '--student-id',
            type=str,
            help='Clean up files for a specific student only',
        )

    def handle(self, *args, **options):
        """Clean up orphaned files and directories"""
        
        dry_run = options['dry_run']
        student_id = options.get('student_id')
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No files will be deleted')
            )
        
        self.stdout.write('Starting file cleanup...')
        
        # 1. Check for orphaned enrollment images
        self.cleanup_orphaned_enrollment_images(dry_run, student_id)
        
        # 2. Check for orphaned enrollment directories
        self.cleanup_orphaned_directories(dry_run, student_id)
        
        # 3. Clean up empty directories
        if not student_id:  # Only do this for full cleanup
            self.cleanup_empty_directories(dry_run)
        
        self.stdout.write(
            self.style.SUCCESS('File cleanup completed!')
        )
    
    def cleanup_orphaned_enrollment_images(self, dry_run=False, student_id=None):
        """Clean up orphaned enrollment image files"""
        try:
            enrollment_images_dir = os.path.join(settings.MEDIA_ROOT, 'enrollment_images')
            
            if not os.path.exists(enrollment_images_dir):
                self.stdout.write('No enrollment images directory found')
                return
            
            # Get all existing image files
            existing_files = []
            for root, dirs, files in os.walk(enrollment_images_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        full_path = os.path.join(root, file)
                        existing_files.append(full_path)
            
            # Get all image paths from database
            db_image_paths = set()
            queryset = EnrollmentImage.objects.all()
            if student_id:
                queryset = queryset.filter(student__student_id=student_id)
                
            for image in queryset:
                if image.image:
                    db_image_paths.add(image.image.path)
            
            # Find orphaned files
            orphaned_files = []
            for file_path in existing_files:
                if file_path not in db_image_paths:
                    # Additional check for student_id if provided
                    if student_id and student_id not in os.path.basename(file_path):
                        continue
                    orphaned_files.append(file_path)
            
            self.stdout.write(f'Found {len(orphaned_files)} orphaned image files')
            
            # Delete or list orphaned files
            deleted_count = 0
            for file_path in orphaned_files:
                if dry_run:
                    self.stdout.write(f'Would delete: {file_path}')
                else:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        self.stdout.write(f'Deleted: {file_path}')
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f'Error deleting {file_path}: {str(e)}')
                        )
            
            if not dry_run and deleted_count > 0:
                self.stdout.write(
                    self.style.SUCCESS(f'Deleted {deleted_count} orphaned image files')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during image cleanup: {str(e)}')
            )
    
    def cleanup_orphaned_directories(self, dry_run=False, student_id=None):
        """Clean up orphaned enrollment directories"""
        try:
            enrollment_data_dir = os.path.join(settings.MEDIA_ROOT, 'enrollment_data')
            
            if not os.path.exists(enrollment_data_dir):
                self.stdout.write('No enrollment data directory found')
                return
            
            # Get all student IDs from database
            db_student_ids = set(Student.objects.values_list('student_id', flat=True))
            
            # Find directories that don't match any student
            orphaned_dirs = []
            for item in os.listdir(enrollment_data_dir):
                dir_path = os.path.join(enrollment_data_dir, item)
                if os.path.isdir(dir_path):
                    # Extract student ID from directory name (format: STUDENTID_name)
                    try:
                        dir_student_id = item.split('_')[0]
                        if dir_student_id not in db_student_ids:
                            # Additional check for specific student_id if provided
                            if student_id and dir_student_id != student_id:
                                continue
                            orphaned_dirs.append(dir_path)
                    except IndexError:
                        # Directory doesn't follow expected format
                        if not student_id:  # Only include in general cleanup
                            orphaned_dirs.append(dir_path)
            
            self.stdout.write(f'Found {len(orphaned_dirs)} orphaned directories')
            
            # Delete or list orphaned directories
            deleted_count = 0
            for dir_path in orphaned_dirs:
                if dry_run:
                    self.stdout.write(f'Would delete directory: {dir_path}')
                else:
                    try:
                        import shutil
                        shutil.rmtree(dir_path)
                        deleted_count += 1
                        self.stdout.write(f'Deleted directory: {dir_path}')
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f'Error deleting {dir_path}: {str(e)}')
                        )
            
            if not dry_run and deleted_count > 0:
                self.stdout.write(
                    self.style.SUCCESS(f'Deleted {deleted_count} orphaned directories')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during directory cleanup: {str(e)}')
            )
    
    def cleanup_empty_directories(self, dry_run=False):
        """Clean up empty directories in media folder"""
        try:
            media_root = settings.MEDIA_ROOT
            empty_dirs = []
            
            for root, dirs, files in os.walk(media_root, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):
                            empty_dirs.append(dir_path)
                    except OSError:
                        pass
            
            self.stdout.write(f'Found {len(empty_dirs)} empty directories')
            
            deleted_count = 0
            for dir_path in empty_dirs:
                if dry_run:
                    self.stdout.write(f'Would delete empty directory: {dir_path}')
                else:
                    try:
                        os.rmdir(dir_path)
                        deleted_count += 1
                        self.stdout.write(f'Deleted empty directory: {dir_path}')
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f'Error deleting {dir_path}: {str(e)}')
                        )
            
            if not dry_run and deleted_count > 0:
                self.stdout.write(
                    self.style.SUCCESS(f'Deleted {deleted_count} empty directories')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during empty directory cleanup: {str(e)}')
            )
