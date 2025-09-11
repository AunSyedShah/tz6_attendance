from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class MainAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main_app'
    
    def ready(self):
        """Import signals when the app is ready"""
        import main_app.signals
        
        # Register camera cleanup signal handlers in main thread
        try:
            from .camera_manager import get_camera_manager
            manager = get_camera_manager()
            manager._setup_signal_handlers()
            logger.info("Camera manager signal handlers registered in app ready")
        except Exception as e:
            logger.warning(f"Could not register camera signal handlers in app ready: {e}")
