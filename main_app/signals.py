"""
Django signals for handling file cleanup on model deletion and camera resource management
"""
from django.db.models.signals import post_delete, pre_delete
from django.dispatch import receiver
from django.conf import settings
from django.core.signals import request_finished
import os
import shutil
import logging
import signal
import sys

from .models import Student, EnrollmentImage

logger = logging.getLogger(__name__)


# Camera cleanup on server shutdown
def cleanup_cameras_on_shutdown(signum, frame):
    """Handle server shutdown and cleanup camera resources"""
    logger.info(f"Server shutdown signal {signum} received, cleaning up cameras...")
    try:
        from .camera_manager import get_camera_manager
        manager = get_camera_manager()
        manager.cleanup_all_cameras()
        logger.info("Camera cleanup completed during server shutdown")
    except Exception as e:
        logger.error(f"Error during camera cleanup on shutdown: {str(e)}")
    
    # Continue with normal shutdown
    sys.exit(0)


# Setup signal handlers for server shutdown
signal.signal(signal.SIGTERM, cleanup_cameras_on_shutdown)
signal.signal(signal.SIGINT, cleanup_cameras_on_shutdown)


@receiver(pre_delete, sender=Student)
def cleanup_student_files(sender, instance, **kwargs):
    """
    Signal handler to clean up files when a student is deleted
    This handles cases where bulk deletion might bypass the model's delete method
    """
    try:
        student_id = instance.student_id
        logger.info(f"Pre-delete cleanup for student {student_id}")
        
        # Get count of images before deletion
        image_count = instance.enrollment_images.count()
        
        # Get enrollment directory path
        enrollment_dir = os.path.join(settings.MEDIA_ROOT, instance.get_enrollment_directory())
        
        # Store info for post-delete cleanup
        instance._cleanup_info = {
            'student_id': student_id,
            'enrollment_dir': enrollment_dir,
            'image_count': image_count
        }
        
    except Exception as e:
        logger.error(f"Error in pre_delete signal for student {instance.student_id}: {str(e)}")


@receiver(post_delete, sender=Student)
def finalize_student_cleanup(sender, instance, **kwargs):
    """
    Signal handler to finalize cleanup after student deletion
    """
    try:
        cleanup_info = getattr(instance, '_cleanup_info', {})
        student_id = cleanup_info.get('student_id', 'unknown')
        enrollment_dir = cleanup_info.get('enrollment_dir')
        
        logger.info(f"Post-delete cleanup for student {student_id}")
        
        # Clean up enrollment directory if it still exists
        if enrollment_dir and os.path.exists(enrollment_dir):
            try:
                shutil.rmtree(enrollment_dir)
                logger.info(f"Deleted enrollment directory: {enrollment_dir}")
            except Exception as e:
                logger.error(f"Error deleting directory {enrollment_dir}: {str(e)}")
        
        # Clean up any orphaned files in enrollment_images directory
        cleanup_orphaned_files(student_id)
        
    except Exception as e:
        logger.error(f"Error in post_delete signal: {str(e)}")


@receiver(post_delete, sender=EnrollmentImage)
def cleanup_enrollment_image_file(sender, instance, **kwargs):
    """
    Signal handler to clean up image files when EnrollmentImage is deleted
    This provides a backup in case the model's delete method doesn't run
    """
    try:
        if instance.image:
            file_path = instance.image.path
            
            # Delete the physical file if it exists
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info(f"Signal: Deleted image file {file_path}")
                
    except Exception as e:
        logger.error(f"Error in post_delete signal for EnrollmentImage {instance.id}: {str(e)}")


def cleanup_orphaned_files(student_id=None):
    """
    Utility function to clean up orphaned files
    """
    try:
        enrollment_images_dir = os.path.join(settings.MEDIA_ROOT, 'enrollment_images')
        
        if not os.path.exists(enrollment_images_dir):
            return
        
        # Get all existing image files
        existing_files = set()
        for root, dirs, files in os.walk(enrollment_images_dir):
            for file in files:
                existing_files.add(os.path.join(root, file))
        
        # Get all image paths from database
        db_image_paths = set()
        for image in EnrollmentImage.objects.all():
            if image.image:
                db_image_paths.add(image.image.path)
        
        # Find orphaned files
        orphaned_files = existing_files - db_image_paths
        
        # Delete orphaned files
        deleted_count = 0
        for file_path in orphaned_files:
            try:
                # Additional check for student_id if provided
                if student_id and student_id not in os.path.basename(file_path):
                    continue
                    
                os.remove(file_path)
                deleted_count += 1
                logger.info(f"Deleted orphaned file: {file_path}")
                
            except Exception as e:
                logger.error(f"Error deleting orphaned file {file_path}: {str(e)}")
        
        if deleted_count > 0:
            logger.info(f"Cleanup completed: {deleted_count} orphaned files deleted")
            
    except Exception as e:
        logger.error(f"Error during orphaned files cleanup: {str(e)}")


def cleanup_empty_directories():
    """
    Utility function to remove empty directories in media folder
    """
    try:
        media_root = settings.MEDIA_ROOT
        
        for root, dirs, files in os.walk(media_root, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    # Try to remove if empty
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        logger.info(f"Removed empty directory: {dir_path}")
                except OSError:
                    # Directory not empty or other error
                    pass
                    
    except Exception as e:
        logger.error(f"Error during empty directory cleanup: {str(e)}")
