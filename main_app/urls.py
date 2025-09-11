from django.urls import path
from . import views
from . import config_views

urlpatterns = [
    # Dashboard
    path('', views.enrollment_dashboard, name='enrollment_dashboard'),
    
    # Student enrollment
    path('enroll/', views.student_enrollment_form, name='student_enrollment_form'),
    path('students/<str:student_id>/', views.student_detail, name='student_detail'),
    path('students/<str:student_id>/capture/', views.enrollment_capture, name='enrollment_capture'),
    path('students/<str:student_id>/demo/', views.demo_enrollment, name='demo_enrollment'),
    
    # Student management
    path('students/<str:student_id>/delete/', views.delete_student, name='delete_student'),
    path('students/<str:student_id>/reset/', views.reset_enrollment, name='reset_enrollment'),
    
    # API endpoints
    path('api/process-frame/', views.process_camera_frame, name='process_camera_frame'),
    path('api/progress/<str:student_id>/', views.get_enrollment_progress, name='get_enrollment_progress'),
    
    # Face Recognition API endpoints (Legacy)
    path('api/train-model/', views.train_recognition_model, name='train_recognition_model'),
    path('api/model-status/', views.recognition_model_status, name='recognition_model_status'),
    path('api/recognize-frame/', views.process_frame_with_recognition, name='process_frame_with_recognition'),
    
    # Deep Learning API endpoints (New)
    path('api/deep/train-model/', views.train_deep_model, name='train_deep_model'),
    path('api/deep/model-status/', views.deep_model_status, name='deep_model_status'),
    path('api/deep/recognize-frame/', views.process_frame_deep_recognition, name='process_frame_deep_recognition'),
    path('api/deep/verify-installation/', views.verify_deep_installation, name='verify_deep_installation'),
    
    # Attendance Management
    path('attendance/', views.attendance_dashboard, name='attendance_dashboard'),
    path('api/attendance/start-session/', views.start_attendance_session, name='start_attendance_session'),
    path('api/attendance/end-session/<int:session_id>/', views.end_attendance_session, name='end_attendance_session'),
    
    # Configuration Management
    path('admin/config-status/', config_views.configuration_status, name='configuration_status'),
    path('api/reload-configuration/', config_views.api_reload_configuration, name='api_reload_configuration'),
    
    # Camera Resource Management
    path('api/camera/cleanup/', views.cleanup_camera_resources, name='cleanup_camera_resources'),
    path('api/camera/status/', views.camera_status, name='camera_status'),
]
