# Camera Resource Management Implementation

## Overview
This document describes the comprehensive camera resource management system implemented for the TZ6 Attendance System. The system ensures proper cleanup of camera resources in all scenarios where the process stops, tabs are closed, or the server is stopped.

## Problem Statement
Previously, camera resources were not properly released when:
- Browser tabs were closed
- Users navigated away from pages
- The Django server was stopped
- Processes were interrupted (Ctrl+C)
- Browser crashes occurred

This led to camera resources remaining locked, preventing other applications from accessing the camera.

## Solution Architecture

### 1. Backend Camera Management (`camera_manager.py`)

#### CameraManager Class
- **Singleton Pattern**: Ensures only one camera manager instance
- **Thread Safety**: Uses threading locks for concurrent access
- **Resource Tracking**: Maintains dictionary of active cameras
- **Session Mapping**: Maps session IDs to camera IDs for cleanup

#### Key Features
- **Signal Handlers**: Registers SIGINT and SIGTERM handlers for graceful shutdown
- **Automatic Cleanup**: Uses `atexit` for Python process termination
- **Django Integration**: Connects to Django's `request_finished` signal
- **Error Handling**: Comprehensive exception handling for all operations

#### API Methods
```python
acquire_camera(camera_id, session_id) -> cv2.VideoCapture
release_camera(camera_id) -> bool
release_session_camera(session_id) -> bool
cleanup_all_cameras() -> None
get_active_cameras() -> Dict
is_camera_active(camera_id) -> bool
```

### 2. Django Signal Integration (`signals.py`)

#### Server Shutdown Handlers
```python
signal.signal(signal.SIGTERM, cleanup_cameras_on_shutdown)
signal.signal(signal.SIGINT, cleanup_cameras_on_shutdown)
```

#### Features
- **Graceful Shutdown**: Properly releases cameras before server exit
- **Logging**: Comprehensive logging of cleanup operations
- **Error Recovery**: Continues shutdown even if cleanup fails

### 3. Management Commands

#### Camera Cleanup Command
```bash
python manage.py cleanup_cameras [--force]
```

**Features:**
- Manual camera resource cleanup
- Force cleanup option for stubborn resources
- Status reporting and error handling

### 4. Frontend JavaScript Enhancements

#### Enhanced Event Handlers
```javascript
// Multiple cleanup scenarios
window.addEventListener('beforeunload', cleanupHandler);
document.addEventListener('visibilitychange', pauseResumeHandler);
window.addEventListener('pagehide', cleanupHandler);
document.addEventListener('keydown', manualStopHandler);
```

#### Camera Stream Management
- **Stream Tracking**: Maintains reference to `MediaStream` object
- **Track Stopping**: Properly stops all media tracks
- **State Management**: Tracks capture states and intervals
- **Server Communication**: Notifies backend of cleanup events

#### Heartbeat System
- **Periodic Check**: 30-second intervals to verify camera status
- **Session Monitoring**: Helps detect and recover from crashes
- **Resource Verification**: Confirms camera availability

### 5. API Endpoints

#### Camera Cleanup Endpoint
```
POST /api/camera/cleanup/
{
    "session_id": "camera_123456789_abc123",
    "reason": "page_unload"
}
```

#### Camera Status Endpoint
```
GET /api/camera/status/
Response: {
    "active_cameras": 1,
    "camera_ids": ["0"],
    "timestamp": "2025-09-11T10:30:00Z"
}
```

## Implementation Details

### 1. Browser Tab Close Detection
```javascript
window.addEventListener('beforeunload', function(event) {
    isPageUnloading = true;
    stopCameraStream();
    notifyServerCameraCleanup('page_unload');
});
```

### 2. Server Process Interruption
```python
def cleanup_cameras_on_shutdown(signum, frame):
    logger.info(f"Server shutdown signal {signum} received")
    manager = get_camera_manager()
    manager.cleanup_all_cameras()
    sys.exit(0)
```

### 3. Page Navigation Detection
```javascript
window.addEventListener('pagehide', function() {
    stopCameraStream();
    notifyServerCameraCleanup('page_hide');
});
```

### 4. Manual Cleanup Shortcut
- **Keyboard Shortcut**: `Ctrl+Shift+C`
- **Visual Feedback**: Status message updates
- **Immediate Effect**: Stops camera stream instantly

## Cleanup Scenarios Covered

### 1. Normal Page Close
- **Trigger**: User clicks close button or navigates away
- **Handler**: `beforeunload` event
- **Action**: Stop camera tracks, notify server, cleanup resources

### 2. Browser Crash
- **Detection**: Heartbeat system failure
- **Recovery**: Server-side timeout and cleanup
- **Fallback**: Next page load reinitializes

### 3. Server Shutdown
- **Trigger**: `SIGTERM` or `SIGINT` signals
- **Handler**: Signal handler function
- **Action**: Cleanup all active cameras, log operations

### 4. Process Interruption
- **Trigger**: `Ctrl+C` or process kill
- **Handler**: `atexit` and signal handlers
- **Action**: Immediate camera release

### 5. Tab Switch/Minimize
- **Trigger**: `visibilitychange` event
- **Action**: Pause processing, maintain camera connection
- **Resume**: Restart processing when tab becomes visible

## Error Handling

### Frontend Error Recovery
```javascript
try {
    cameraStream.getTracks().forEach(track => track.stop());
} catch (error) {
    console.error('Error stopping camera stream:', error);
    // Continue with other cleanup steps
}
```

### Backend Error Recovery
```python
try:
    cap.release()
    logger.info(f"Camera {camera_id} released successfully")
except Exception as e:
    logger.error(f"Error releasing camera: {str(e)}")
    # Continue with cleanup of other cameras
```

## Performance Considerations

### 1. Resource Efficiency
- **Singleton Pattern**: Single camera manager instance
- **Lazy Loading**: Cameras acquired only when needed
- **Immediate Release**: Resources freed as soon as possible

### 2. Network Efficiency
- **Batched Notifications**: Combine multiple cleanup calls
- **Heartbeat Optimization**: 30-second intervals to minimize traffic
- **Error Tolerance**: Network failures don't block cleanup

### 3. Memory Management
- **Reference Cleanup**: Clear all object references
- **Event Listener Removal**: Prevent memory leaks
- **Interval Clearing**: Stop all timers and intervals

## Testing and Validation

### 1. Manual Testing Scenarios
- [ ] Close browser tab during enrollment
- [ ] Navigate away from enrollment page
- [ ] Stop Django server with Ctrl+C
- [ ] Kill Django process with SIGTERM
- [ ] Refresh page during camera use
- [ ] Switch tabs during enrollment
- [ ] Use manual stop shortcut (Ctrl+Shift+C)

### 2. Automated Testing
```bash
# Test management command
python manage.py cleanup_cameras

# Test API endpoints
curl -X GET http://localhost:8000/api/camera/status/
curl -X POST http://localhost:8000/api/camera/cleanup/ -d '{"reason":"test"}'
```

### 3. Resource Verification
```bash
# Check if camera is available after cleanup
lsof | grep video  # Linux
# Should show no Python processes holding camera
```

## Troubleshooting

### Common Issues

#### 1. Camera Still Locked After Cleanup
**Cause**: Hardware driver issue or process crash
**Solution**: 
```bash
# Manual camera release (Linux)
sudo fuser -k /dev/video0
# Or restart camera service
sudo service uvcvideo restart
```

#### 2. Cleanup Not Triggered
**Cause**: JavaScript event handlers not registered
**Solution**: Check browser console for errors, verify event handler setup

#### 3. Server Cleanup Fails
**Cause**: Signal handlers not registered properly
**Solution**: Check `apps.py` imports signals module correctly

### Debugging Tools

#### 1. Browser Console Monitoring
```javascript
// Enable debug logging
localStorage.setItem('camera_debug', 'true');
```

#### 2. Server Log Analysis
```bash
# Monitor camera-related logs
tail -f logs/django.log | grep -i camera
```

#### 3. Resource Monitoring
```bash
# Check active camera processes
ps aux | grep python | grep camera
lsof | grep video
```

## Benefits

### 1. Resource Management
- **Prevents Camera Lock**: Ensures camera is available for other apps
- **Memory Efficiency**: Proper cleanup prevents memory leaks
- **Process Stability**: Reduces system resource contention

### 2. User Experience
- **Reliable Operation**: Camera consistently available
- **Quick Recovery**: Fast restart after interruptions
- **Visual Feedback**: Clear status messages and indicators

### 3. System Stability
- **Graceful Degradation**: System continues working even with errors
- **Error Recovery**: Automatic retry and fallback mechanisms
- **Monitoring**: Comprehensive logging and status reporting

## Future Enhancements

### 1. Advanced Features
- **Multi-Camera Support**: Handle multiple cameras simultaneously
- **Camera Quality Detection**: Automatically select best available camera
- **Hot-Swap Support**: Handle camera disconnect/reconnect

### 2. Monitoring Improvements
- **Dashboard Integration**: Real-time camera status in admin panel
- **Alert System**: Notifications for camera resource issues
- **Performance Metrics**: Track camera usage and performance

### 3. Platform Support
- **Cross-Platform**: Better support for Windows/macOS camera drivers
- **Mobile Devices**: Enhanced support for mobile camera access
- **WebRTC Integration**: Advanced camera stream management

## Conclusion

This comprehensive camera resource management system ensures that camera resources are properly released in all scenarios where the process stops, tabs are closed, or the server is stopped. The implementation covers both frontend and backend components with robust error handling and multiple cleanup strategies for maximum reliability.
