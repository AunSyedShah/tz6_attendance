"""
Middleware for dynamic configuration management and request rate limiting
"""
from django.conf import settings
from django.http import JsonResponse
import logging
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DynamicConfigurationMiddleware:
    """
    Middleware to apply dynamic configuration settings on each request
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Apply dynamic settings before processing request
        self.apply_dynamic_settings()
        
        response = self.get_response(request)
        return response
    
    def apply_dynamic_settings(self):
        """Apply dynamic settings to Django settings"""
        try:
            from main_app.config_service import config_service
            
            # Update file upload settings
            upload_settings = config_service.get_upload_settings()
            settings.FILE_UPLOAD_MAX_MEMORY_SIZE = upload_settings['FILE_UPLOAD_MAX_MEMORY_SIZE']
            settings.DATA_UPLOAD_MAX_MEMORY_SIZE = upload_settings['DATA_UPLOAD_MAX_MEMORY_SIZE']
            
        except Exception as e:
            logger.warning(f"Failed to apply dynamic settings: {str(e)}")
            # Continue with default settings if dynamic config fails
            pass


class FrameProcessingRateLimitMiddleware:
    """
    Middleware to handle rate limiting for 30 FPS frame processing
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.client_requests = defaultdict(lambda: deque())
        self.max_requests_per_second = 35  # Allow slightly more than 30 FPS
        self.cleanup_interval = 60  # Clean old records every minute
        self.last_cleanup = time.time()
    
    def __call__(self, request):
        # Only apply rate limiting to frame processing endpoints
        if self.should_rate_limit(request):
            if not self.is_request_allowed(request):
                return JsonResponse({
                    'error': 'Rate limit exceeded for 30 FPS processing',
                    'fps_limit': self.max_requests_per_second,
                    'retry_after': 1.0 / self.max_requests_per_second
                }, status=429)
        
        response = self.get_response(request)
        return response
    
    def should_rate_limit(self, request):
        """Check if this request should be rate limited"""
        return (
            request.path == '/api/process-frame/' or 
            request.path.endswith('/process-frame/') or
            'process-frame' in request.path
        )
    
    def is_request_allowed(self, request):
        """Check if request is within rate limits"""
        try:
            # Get client identifier (IP address)
            client_ip = self.get_client_ip(request)
            current_time = time.time()
            
            # Clean old requests periodically
            if current_time - self.last_cleanup > self.cleanup_interval:
                self.cleanup_old_requests(current_time)
                self.last_cleanup = current_time
            
            # Get request history for this client
            client_history = self.client_requests[client_ip]
            
            # Remove requests older than 1 second
            while client_history and current_time - client_history[0] > 1.0:
                client_history.popleft()
            
            # Check if client has exceeded rate limit
            if len(client_history) >= self.max_requests_per_second:
                logger.warning(f"Rate limit exceeded for client {client_ip}: {len(client_history)} requests/second")
                return False
            
            # Record this request
            client_history.append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Error in rate limiting: {str(e)}")
            return True  # Allow request if rate limiting fails
    
    def get_client_ip(self, request):
        """Get client IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
    
    def cleanup_old_requests(self, current_time):
        """Clean up old request records to prevent memory leaks"""
        clients_to_remove = []
        
        for client_ip, history in self.client_requests.items():
            # Remove old requests
            while history and current_time - history[0] > 60:  # Keep only last minute
                history.popleft()
            
            # Remove empty histories
            if not history:
                clients_to_remove.append(client_ip)
        
        for client_ip in clients_to_remove:
            del self.client_requests[client_ip]
        
        if clients_to_remove:
            logger.info(f"Cleaned up {len(clients_to_remove)} inactive client rate limit records")
