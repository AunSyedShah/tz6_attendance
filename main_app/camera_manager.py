"""
Camera Resource Manager for TZ6 Attendance System
Handles proper cleanup of camera resources across different scenarios
"""
import cv2
import threading
import signal
import sys
import logging
from typing import Dict, Optional
from django.core.signals import request_finished
from django.dispatch import receiver
import atexit

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Singleton class to manage camera resources and ensure proper cleanup
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CameraManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.active_cameras: Dict[str, cv2.VideoCapture] = {}
        self.session_cameras: Dict[str, str] = {}  # session_id -> camera_id mapping
        self._lock = threading.Lock()
        self._setup_cleanup_handlers()
        self._initialized = True
        
        logger.info("CameraManager initialized")
    
    def _setup_cleanup_handlers(self):
        """Setup various cleanup handlers for different shutdown scenarios"""
        # Handle normal Python exit
        atexit.register(self.cleanup_all_cameras)
        
        # Handle system signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Handle Django request completion
        request_finished.connect(self._cleanup_session_cameras)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, cleaning up cameras...")
        self.cleanup_all_cameras()
        sys.exit(0)
    
    def acquire_camera(self, camera_id: str = "0", session_id: Optional[str] = None) -> Optional[cv2.VideoCapture]:
        """
        Acquire a camera resource
        
        Args:
            camera_id: Camera identifier (default: "0" for default camera)
            session_id: Session identifier for tracking
            
        Returns:
            cv2.VideoCapture object or None if acquisition failed
        """
        with self._lock:
            try:
                # Check if camera is already in use
                if camera_id in self.active_cameras:
                    logger.warning(f"Camera {camera_id} is already in use")
                    return self.active_cameras[camera_id]
                
                # Try to open camera
                cap = cv2.VideoCapture(int(camera_id) if camera_id.isdigit() else camera_id)
                
                if not cap.isOpened():
                    logger.error(f"Failed to open camera {camera_id}")
                    return None
                
                # Store camera reference
                self.active_cameras[camera_id] = cap
                
                # Map session to camera if session_id provided
                if session_id:
                    self.session_cameras[session_id] = camera_id
                
                logger.info(f"Camera {camera_id} acquired successfully" + 
                           (f" for session {session_id}" if session_id else ""))
                
                return cap
                
            except Exception as e:
                logger.error(f"Error acquiring camera {camera_id}: {str(e)}")
                return None
    
    def release_camera(self, camera_id: str) -> bool:
        """
        Release a specific camera
        
        Args:
            camera_id: Camera identifier to release
            
        Returns:
            True if released successfully, False otherwise
        """
        with self._lock:
            try:
                if camera_id in self.active_cameras:
                    cap = self.active_cameras[camera_id]
                    cap.release()
                    del self.active_cameras[camera_id]
                    
                    # Remove from session mapping
                    sessions_to_remove = [sid for sid, cid in self.session_cameras.items() if cid == camera_id]
                    for session_id in sessions_to_remove:
                        del self.session_cameras[session_id]
                    
                    logger.info(f"Camera {camera_id} released successfully")
                    return True
                else:
                    logger.warning(f"Camera {camera_id} not found in active cameras")
                    return False
                    
            except Exception as e:
                logger.error(f"Error releasing camera {camera_id}: {str(e)}")
                return False
    
    def release_session_camera(self, session_id: str) -> bool:
        """
        Release camera associated with a specific session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if released successfully, False otherwise
        """
        with self._lock:
            if session_id in self.session_cameras:
                camera_id = self.session_cameras[session_id]
                return self.release_camera(camera_id)
            return False
    
    def cleanup_all_cameras(self):
        """Release all active cameras"""
        with self._lock:
            try:
                camera_ids = list(self.active_cameras.keys())
                for camera_id in camera_ids:
                    try:
                        cap = self.active_cameras[camera_id]
                        cap.release()
                        logger.info(f"Camera {camera_id} released during cleanup")
                    except Exception as e:
                        logger.error(f"Error releasing camera {camera_id} during cleanup: {str(e)}")
                
                self.active_cameras.clear()
                self.session_cameras.clear()
                
                # Also call cv2.destroyAllWindows() to ensure all OpenCV windows are closed
                cv2.destroyAllWindows()
                
                logger.info("All cameras cleaned up successfully")
                
            except Exception as e:
                logger.error(f"Error during camera cleanup: {str(e)}")
    
    def get_active_cameras(self) -> Dict[str, cv2.VideoCapture]:
        """Get dictionary of currently active cameras"""
        with self._lock:
            return self.active_cameras.copy()
    
    def is_camera_active(self, camera_id: str) -> bool:
        """Check if a specific camera is currently active"""
        with self._lock:
            return camera_id in self.active_cameras
    
    @receiver(request_finished)
    def _cleanup_session_cameras(self, sender, **kwargs):
        """Cleanup cameras when Django request finishes"""
        # This is a simple cleanup - in a real application you might want
        # to track sessions more carefully
        pass


# Global camera manager instance
camera_manager = CameraManager()


def get_camera_manager() -> CameraManager:
    """Get the global camera manager instance"""
    return camera_manager


# Django management command helper
def cleanup_cameras_command():
    """Helper function for management commands to cleanup cameras"""
    try:
        manager = get_camera_manager()
        manager.cleanup_all_cameras()
        logger.info("Camera cleanup completed via management command")
        return True
    except Exception as e:
        logger.error(f"Error in camera cleanup command: {str(e)}")
        return False
