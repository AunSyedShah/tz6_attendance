"""
Django management command to cleanup camera resources
Usage: python manage.py cleanup_cameras
"""
from django.core.management.base import BaseCommand
from main_app.camera_manager import cleanup_cameras_command
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Cleanup all active camera resources'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force cleanup even if cameras appear to be in use',
        )
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE('Starting camera cleanup...'))
        
        try:
            success = cleanup_cameras_command()
            
            if success:
                self.stdout.write(
                    self.style.SUCCESS('Successfully cleaned up all camera resources')
                )
            else:
                self.stdout.write(
                    self.style.ERROR('Camera cleanup completed with some errors - check logs')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Camera cleanup failed: {str(e)}')
            )
            logger.error(f"Camera cleanup command failed: {str(e)}")
