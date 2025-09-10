from django.urls import path
from . import views

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
    
    # Face Recognition API endpoints
    path('api/train-model/', views.train_recognition_model, name='train_recognition_model'),
    path('api/model-status/', views.recognition_model_status, name='recognition_model_status'),
    path('api/recognize-frame/', views.process_frame_with_recognition, name='process_frame_with_recognition'),
    
    # Attendance Management
    path('attendance/', views.attendance_dashboard, name='attendance_dashboard'),
    path('api/attendance/start-session/', views.start_attendance_session, name='start_attendance_session'),
    path('api/attendance/end-session/<int:session_id>/', views.end_attendance_session, name='end_attendance_session'),
]
