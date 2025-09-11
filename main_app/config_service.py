"""
Configuration service for dynamic system settings
"""
from typing import Any, Dict, Union
from django.core.cache import cache
from django.conf import settings as django_settings
import json
import logging

logger = logging.getLogger(__name__)


class ConfigurationService:
    """Service for managing dynamic system configuration"""
    
    CACHE_PREFIX = 'system_config_'
    CACHE_TIMEOUT = 300  # 5 minutes
    
    @classmethod
    def get_setting(cls, key: str, default: Any = None, use_cache: bool = True) -> Any:
        """
        Get a configuration setting by key
        
        Args:
            key: Setting key
            default: Default value if setting not found
            use_cache: Whether to use caching
            
        Returns:
            Setting value or default
        """
        cache_key = f"{cls.CACHE_PREFIX}{key}"
        
        # Try cache first
        if use_cache:
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
        
        try:
            from .models import SystemConfiguration
            setting = SystemConfiguration.objects.get(key=key, is_active=True)
            value = setting.get_value()
            
            # Cache the value
            if use_cache:
                cache.set(cache_key, value, cls.CACHE_TIMEOUT)
                
            return value
            
        except SystemConfiguration.DoesNotExist:
            logger.warning(f"Setting '{key}' not found, using default: {default}")
            return default
        except Exception as e:
            logger.error(f"Error getting setting '{key}': {str(e)}")
            return default
    
    @classmethod
    def set_setting(cls, key: str, value: Any, modified_by: str = None) -> bool:
        """
        Set a configuration setting
        
        Args:
            key: Setting key
            value: New value
            modified_by: User who made the change
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from .models import SystemConfiguration
            setting = SystemConfiguration.objects.get(key=key)
            setting.set_value(value, modified_by)
            
            # Clear cache
            cache_key = f"{cls.CACHE_PREFIX}{key}"
            cache.delete(cache_key)
            
            logger.info(f"Setting '{key}' updated to '{value}' by {modified_by or 'system'}")
            return True
            
        except SystemConfiguration.DoesNotExist:
            logger.error(f"Setting '{key}' not found")
            return False
        except Exception as e:
            logger.error(f"Error setting '{key}': {str(e)}")
            return False
    
    @classmethod
    def get_enrollment_settings(cls) -> Dict[str, Any]:
        """Get all enrollment-related settings"""
        return {
            'TARGET_IMAGES_TOTAL': cls.get_setting('TARGET_IMAGES_TOTAL', 100),
            'IMAGES_PER_PROMPT': cls.get_setting('IMAGES_PER_PROMPT', 15),
            'FACE_SIZE_THRESHOLD': tuple(cls.get_setting('FACE_SIZE_THRESHOLD', [50, 50])),
            'IMAGE_SIZE': tuple(cls.get_setting('IMAGE_SIZE', [128, 128])),
            'BLUR_THRESHOLD': cls.get_setting('BLUR_THRESHOLD', 80),
            'BRIGHTNESS_RANGE': (
                cls.get_setting('BRIGHTNESS_RANGE_MIN', 15),
                cls.get_setting('BRIGHTNESS_RANGE_MAX', 230)
            ),
            'LOW_LIGHT_THRESHOLD': cls.get_setting('LOW_LIGHT_THRESHOLD', 50),
        }
    
    @classmethod
    def get_upload_settings(cls) -> Dict[str, Any]:
        """Get all file upload settings"""
        return {
            'FILE_UPLOAD_MAX_MEMORY_SIZE': cls.get_setting('FILE_UPLOAD_MAX_MEMORY_SIZE', 10) * 1024 * 1024,
            'DATA_UPLOAD_MAX_MEMORY_SIZE': cls.get_setting('DATA_UPLOAD_MAX_MEMORY_SIZE', 10) * 1024 * 1024,
        }
    
    @classmethod
    def get_recognition_settings(cls) -> Dict[str, Any]:
        """Get all face recognition settings"""
        return {
            'CONFIDENCE_THRESHOLD': cls.get_setting('RECOGNITION_CONFIDENCE_THRESHOLD', 0.7),
            'ATTENDANCE_THRESHOLD': cls.get_setting('ATTENDANCE_AUTO_MARK_THRESHOLD', 0.85),
            'MODEL_NAME': cls.get_setting('DEEP_LEARNING_MODEL', 'Facenet512'),
        }
    
    @classmethod
    def get_performance_settings(cls) -> Dict[str, Any]:
        """Get all performance settings"""
        return {
            'CAMERA_FPS': cls.get_setting('CAMERA_FPS', 30),
            'PROCESSING_TIMEOUT': cls.get_setting('PROCESSING_TIMEOUT', 30),
        }
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Dict[str, Any]]:
        """Get all settings grouped by category"""
        return {
            'enrollment': cls.get_enrollment_settings(),
            'upload': cls.get_upload_settings(),
            'recognition': cls.get_recognition_settings(),
            'performance': cls.get_performance_settings(),
        }
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached settings"""
        try:
            from .models import SystemConfiguration
            all_keys = SystemConfiguration.objects.values_list('key', flat=True)
            
            for key in all_keys:
                cache_key = f"{cls.CACHE_PREFIX}{key}"
                cache.delete(cache_key)
                
            logger.info("Configuration cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing configuration cache: {str(e)}")
    
    @classmethod
    def export_settings(cls) -> Dict[str, Any]:
        """Export all settings for backup"""
        try:
            from .models import SystemConfiguration
            settings = {}
            
            for config in SystemConfiguration.objects.filter(is_active=True):
                settings[config.key] = {
                    'value': config.get_value(),
                    'name': config.name,
                    'category': config.category,
                    'description': config.description,
                    'type': config.setting_type,
                }
            
            return settings
            
        except Exception as e:
            logger.error(f"Error exporting settings: {str(e)}")
            return {}
    
    @classmethod
    def import_settings(cls, settings_data: Dict[str, Any], modified_by: str = None) -> int:
        """Import settings from backup data"""
        imported_count = 0
        
        try:
            for key, data in settings_data.items():
                if cls.set_setting(key, data.get('value'), modified_by):
                    imported_count += 1
                    
            cls.clear_cache()
            logger.info(f"Imported {imported_count} settings")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing settings: {str(e)}")
            return 0


# Global configuration service instance
config_service = ConfigurationService()
