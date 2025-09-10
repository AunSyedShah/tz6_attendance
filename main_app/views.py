from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.utils import timezone
from django.conf import settings
import json
import logging
import cv2
import numpy as np
from PIL import Image
import io
import base64

from .models import Student, EnrollmentImage, EnrollmentSession, AttendanceRecord, AttendanceSession, RecognitionLog
from .services import face_detection_service
from .face_recognition_service import face_recognition_service

logger = logging.getLogger(__name__)


def make_json_serializable(obj):
    """
    Convert NumPy types to native Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def enrollment_dashboard(request):
    """Main dashboard for student enrollment"""
    
    # Get all students with enrollment statistics
    students = Student.objects.annotate(
        image_count=Count('enrollment_images')
    ).order_by('-enrollment_date')
    
    # Calculate progress percentage for each student
    for student in students:
        student.progress_percentage = min(100, (student.total_images_captured / 100) * 100)
    
    # Pagination
    paginator = Paginator(students, 10)  # Show 10 students per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Statistics
    stats = {
        'total_students': Student.objects.count(),
        'completed_enrollments': Student.objects.filter(enrollment_completed=True).count(),
        'active_students': Student.objects.filter(is_active=True).count(),
        'total_images': EnrollmentImage.objects.count(),
    }
    
    context = {
        'page_obj': page_obj,
        'stats': stats,
        'enrollment_prompts': face_detection_service.enrollment_prompts,
    }
    
    return render(request, 'main_app/dashboard.html', context)


def student_enrollment_form(request):
    """Display student enrollment form"""
    
    if request.method == 'POST':
        student_name = request.POST.get('student_name', '').strip()
        student_id = request.POST.get('student_id', '').strip().upper()
        email = request.POST.get('email', '').strip()
        phone = request.POST.get('phone', '').strip()
        
        # Validation
        if not student_name or not student_id:
            messages.error(request, 'Student name and ID are required.')
            return render(request, 'main_app/enrollment_form.html')
        
        # Check if student already exists
        if Student.objects.filter(student_id=student_id).exists():
            messages.error(request, f'Student with ID {student_id} already exists.')
            return render(request, 'main_app/enrollment_form.html')
        
        # Create new student
        student = Student.objects.create(
            student_id=student_id,
            full_name=student_name,
            email=email if email else None,
            phone=phone if phone else None
        )
        
        messages.success(request, f'Student {student_name} created successfully!')
        return redirect('enrollment_capture', student_id=student.student_id)
    
    return render(request, 'main_app/enrollment_form.html')


def enrollment_capture(request, student_id):
    """Main enrollment capture interface"""
    
    student = get_object_or_404(Student, student_id=student_id)
    
    # Get enrollment progress
    progress = face_detection_service.get_enrollment_progress(student)
    
    # Create or get current enrollment session
    session = EnrollmentSession.objects.filter(
        student=student,
        session_completed=False
    ).first()
    
    if not session:
        session = EnrollmentSession.objects.create(
            student=student,
            session_mode='live',
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            ip_address=request.META.get('REMOTE_ADDR')
        )
    
    context = {
        'student': student,
        'session': session,
        'progress': progress,
        'enrollment_prompts': face_detection_service.enrollment_prompts,
        'target_images_total': face_detection_service.target_images_total,
        'images_per_prompt': face_detection_service.images_per_prompt,
    }
    
    return render(request, 'main_app/enrollment_capture.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def process_camera_frame(request):
    """
    Process camera frame for face detection and capture
    Optimized for 30 FPS video stream processing
    Enhanced with optional face recognition for attendance
    """
    
    try:
        data = json.loads(request.body)
        student_id = data.get('student_id')
        image_data = data.get('image_data')
        capture_image = data.get('capture_image', False)
        frame_number = data.get('frame_number', 0)  # For 30 FPS frame tracking
        recognition_mode = data.get('recognition_mode', False)  # Enable face recognition
        
        if not student_id or not image_data:
            return JsonResponse({'error': 'Missing required data'}, status=400)

        student = get_object_or_404(Student, student_id=student_id)
        
        # For 30 FPS optimization: Process every frame for face detection,
        # but only capture high-quality frames when requested
        frame = face_detection_service.process_image_from_base64(image_data)
        if frame is None:
            return JsonResponse({'error': 'Invalid image data'}, status=400)

        # Step 1: Capturing Video Frames (30 FPS)
        # Convert to grayscale for processing efficiency
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Face Detection - Real-time face detection on every frame
        face_detected, face_coords, status_msg, confidence, quality_metrics = \
            face_detection_service.detect_and_validate_face(frame)
        
        response_data = {
            'face_detected': face_detected,
            'status_message': status_msg,
            'confidence': confidence,
            'face_coordinates': face_coords,
            'quality_metrics': quality_metrics,
            'frame_number': frame_number,
            'fps_info': '30 FPS Processing Active'
        }
        
        # Optional: Face Recognition for attendance (if recognition_mode is enabled)
        if recognition_mode and face_detected and frame is not None:
            try:
                recognized_student_id, recognition_confidence, recognized_name = face_recognition_service.recognize_face(frame)
                response_data.update({
                    'recognition_enabled': True,
                    'recognized_student_id': recognized_student_id,
                    'recognized_name': recognized_name,
                    'recognition_confidence': recognition_confidence,
                    'recognition_successful': recognized_student_id is not None
                })
                
                # Auto-mark attendance if confidence is high
                if recognized_student_id and recognition_confidence >= 0.8:
                    try:
                        recognized_student = Student.objects.get(student_id=recognized_student_id)
                        today = timezone.now().date()
                        
                        attendance_record, created = AttendanceRecord.objects.get_or_create(
                            student=recognized_student,
                            date=today,
                            defaults={
                                'recognition_confidence': recognition_confidence,
                                'recognition_method': 'face_recognition'
                            }
                        )
                        
                        if created or not attendance_record.entry_time:
                            attendance_record.mark_entry(recognition_confidence)
                            response_data['attendance_auto_marked'] = True
                            logger.info(f"Auto-marked attendance for {recognized_name}")
                        
                    except Student.DoesNotExist:
                        logger.warning(f"Recognized student {recognized_student_id} not found in database")
                        
            except Exception as e:
                logger.error(f"Error in face recognition: {str(e)}")
                response_data.update({
                    'recognition_enabled': True,
                    'recognition_error': str(e)
                })
        
        # If face is detected and capture is requested
        if face_detected and capture_image and face_coords:
            try:
                # Get current enrollment progress
                progress = face_detection_service.get_enrollment_progress(student)
                current_prompt_index = progress['current_prompt_index']
                
                # Count images for current prompt
                current_prompt_images = EnrollmentImage.objects.filter(
                    student=student,
                    prompt_index=current_prompt_index
                ).count()
                
                # Check if current prompt is complete
                if current_prompt_images >= face_detection_service.images_per_prompt:
                    return JsonResponse(make_json_serializable({
                        **response_data,
                        'prompt_complete': True,
                        'message': 'Current prompt completed, moving to next...'
                    }))
                
                # Process and save image
                processed_image = face_detection_service.preprocess_face_image(frame, face_coords)
                
                # Generate unique sequence number using current timestamp microseconds
                import time
                unique_sequence = int((time.time() * 1000000) % 100000)  # Use microseconds for uniqueness
                
                # Save to database (no unique constraint issues now)
                enrollment_image = face_detection_service.save_processed_image(
                    processed_image=processed_image,
                    student=student,
                    prompt_index=current_prompt_index,
                    image_sequence=unique_sequence,
                    face_coords=face_coords,
                    quality_metrics=quality_metrics,
                    capture_mode='live'
                )
                
                
                # Update student's total image count
                student.total_images_captured = EnrollmentImage.objects.filter(student=student).count()
                
                # Check if enrollment is complete
                if student.total_images_captured >= face_detection_service.target_images_total:
                    student.enrollment_completed = True
                
                student.save()
                
                # Update session statistics
                session = EnrollmentSession.objects.filter(
                    student=student,
                    session_completed=False
                ).first()
                
                if session:
                    session.successful_captures += 1
                    session.total_attempts += 1
                    session.save()
                
                # Get updated progress
                updated_progress = face_detection_service.get_enrollment_progress(student)
                
                response_data.update({
                    'image_captured': True,
                    'image_id': enrollment_image.id,
                    'total_images': student.total_images_captured,
                    'enrollment_progress': updated_progress,
                    'enrollment_completed': student.enrollment_completed,
                })
                
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                response_data['error'] = f"Failed to save image: {str(e)}"
                
                # Update session with failed capture
                session = EnrollmentSession.objects.filter(
                    student=student,
                    session_completed=False
                ).first()
                
                if session:
                    session.failed_captures += 1
                    session.total_attempts += 1
                    session.save()
        
        return JsonResponse(make_json_serializable(response_data))
        
    except Exception as e:
        logger.error(f"Error processing camera frame: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_enrollment_progress(request, student_id):
    """Get current enrollment progress for a student"""
    
    try:
        student = get_object_or_404(Student, student_id=student_id)
        progress = face_detection_service.get_enrollment_progress(student)
        
        return JsonResponse(make_json_serializable({
            'success': True,
            'progress': progress
        }))
        
    except Exception as e:
        logger.error(f"Error getting enrollment progress: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def demo_enrollment(request, student_id):
    """Demo enrollment using file uploads"""
    
    student = get_object_or_404(Student, student_id=student_id)
    
    if request.method == 'POST' and request.FILES.getlist('images'):
        uploaded_files = request.FILES.getlist('images')
        successful_uploads = 0
        failed_uploads = 0
        errors = []
        
        # Create or get demo session
        session = EnrollmentSession.objects.filter(
            student=student,
            session_completed=False,
            session_mode='demo'
        ).first()
        
        if not session:
            session = EnrollmentSession.objects.create(
                student=student,
                session_mode='demo',
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
                ip_address=request.META.get('REMOTE_ADDR')
            )
        
        for uploaded_file in uploaded_files:
            try:
                # Load and process image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Convert to OpenCV format
                if len(image_array.shape) == 3:
                    if image_array.shape[2] == 3:  # RGB
                        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    elif image_array.shape[2] == 4:  # RGBA
                        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                    else:
                        raise ValueError("Unsupported image format")
                else:
                    image_cv = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                
                # Face detection and validation
                face_detected, face_coords, status_msg, confidence, quality_metrics = \
                    face_detection_service.detect_and_validate_face(image_cv)
                
                session.total_attempts += 1
                
                if face_detected and face_coords:
                    # Process and save image
                    processed_image = face_detection_service.preprocess_face_image(image_cv, face_coords)
                    
                    # Use current number of images as prompt index (distribute across prompts)
                    current_total = EnrollmentImage.objects.filter(student=student).count()
                    prompt_index = (current_total // face_detection_service.images_per_prompt) % len(face_detection_service.enrollment_prompts)
                    image_sequence = current_total % face_detection_service.images_per_prompt
                    
                    # Save to database
                    face_detection_service.save_processed_image(
                        processed_image=processed_image,
                        student=student,
                        prompt_index=prompt_index,
                        image_sequence=image_sequence,
                        face_coords=face_coords,
                        quality_metrics=quality_metrics,
                        capture_mode='demo'
                    )
                    
                    successful_uploads += 1
                    session.successful_captures += 1
                else:
                    failed_uploads += 1
                    session.failed_captures += 1
                    errors.append(f"{uploaded_file.name}: {status_msg}")
                
            except Exception as e:
                failed_uploads += 1
                session.failed_captures += 1
                errors.append(f"{uploaded_file.name}: {str(e)}")
        
        # Update student and session
        student.total_images_captured = EnrollmentImage.objects.filter(student=student).count()
        if student.total_images_captured >= face_detection_service.target_images_total:
            student.enrollment_completed = True
            session.session_completed = True
            session.session_end = timezone.now()
        
        student.save()
        session.save()
        
        # Messages
        if successful_uploads > 0:
            messages.success(request, f'Successfully processed {successful_uploads} images.')
        if failed_uploads > 0:
            messages.warning(request, f'{failed_uploads} images failed processing.')
        for error in errors[:5]:  # Show only first 5 errors
            messages.error(request, error)
    
    # Get current progress
    progress = face_detection_service.get_enrollment_progress(student)
    
    context = {
        'student': student,
        'progress': progress,
        'enrollment_prompts': face_detection_service.enrollment_prompts,
    }
    
    return render(request, 'main_app/demo_enrollment.html', context)


def student_detail(request, student_id):
    """Display detailed information about a student's enrollment"""
    
    student = get_object_or_404(Student, student_id=student_id)
    
    # Get enrollment images grouped by prompt
    enrollment_images = EnrollmentImage.objects.filter(student=student).order_by('prompt_index', 'image_sequence')
    
    # Group images by prompt with prompt text
    images_by_prompt = {}
    for image in enrollment_images:
        if image.prompt_index not in images_by_prompt:
            # Get prompt text from the service
            prompt_text = "Unknown Prompt"
            if image.prompt_index < len(face_detection_service.enrollment_prompts):
                prompt_text = face_detection_service.enrollment_prompts[image.prompt_index]
            
            images_by_prompt[image.prompt_index] = {
                'prompt_text': prompt_text,
                'images': []
            }
        images_by_prompt[image.prompt_index]['images'].append(image)
    
    # Get sessions
    sessions = EnrollmentSession.objects.filter(student=student).order_by('-session_start')
    
    # Get progress
    progress = face_detection_service.get_enrollment_progress(student)
    
    context = {
        'student': student,
        'images_by_prompt': images_by_prompt,
        'sessions': sessions,
        'progress': progress,
        'enrollment_prompts': face_detection_service.enrollment_prompts,
    }
    
    return render(request, 'main_app/student_detail.html', context)


@require_http_methods(["POST"])
def delete_student(request, student_id):
    """Delete a student and all associated data"""
    
    try:
        student = get_object_or_404(Student, student_id=student_id)
        student_name = student.full_name
        
        # Delete student (cascade will delete related images and sessions)
        student.delete()
        
        messages.success(request, f'Student {student_name} and all associated data deleted successfully.')
        return redirect('enrollment_dashboard')
        
    except Exception as e:
        logger.error(f"Error deleting student: {str(e)}")
        messages.error(request, f'Error deleting student: {str(e)}')
        return redirect('enrollment_dashboard')


@require_http_methods(["POST"])
def reset_enrollment(request, student_id):
    """Reset a student's enrollment data"""
    
    try:
        student = get_object_or_404(Student, student_id=student_id)
        
        # Delete all enrollment images and sessions
        EnrollmentImage.objects.filter(student=student).delete()
        EnrollmentSession.objects.filter(student=student).delete()
        
        # Reset student status
        student.enrollment_completed = False
        student.total_images_captured = 0
        student.save()
        
        messages.success(request, f'Enrollment data reset for {student.full_name}.')
        return redirect('student_detail', student_id=student_id)
        
    except Exception as e:
        logger.error(f"Error resetting enrollment: {str(e)}")
        messages.error(request, f'Error resetting enrollment: {str(e)}')
        return redirect('student_detail', student_id=student_id)


# ==========================================
# FACE RECOGNITION AND ATTENDANCE VIEWS
# ==========================================

@csrf_exempt
@require_http_methods(["POST"])
def train_recognition_model(request):
    """
    Train the face recognition model with all enrolled students
    """
    try:
        logger.info("Starting face recognition model training")
        
        # Check if there are enough students
        student_count = Student.objects.filter(enrollment_completed=True).count()
        if student_count < 2:
            return JsonResponse({
                'success': False,
                'error': 'At least 2 enrolled students required for training',
                'student_count': student_count
            })
        
        # Train the model
        success = face_recognition_service.train_recognition_model()
        
        if success:
            # Get model statistics
            stats = face_recognition_service.get_model_statistics()
            
            logger.info(f"Face recognition model trained successfully: {stats}")
            
            return JsonResponse({
                'success': True,
                'message': 'Face recognition model trained successfully',
                'statistics': stats
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to train face recognition model'
            })
            
    except Exception as e:
        logger.error(f"Error training recognition model: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Error training model: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["GET"])
def recognition_model_status(request):
    """
    Get the current status of the face recognition model
    """
    try:
        stats = face_recognition_service.get_model_statistics()
        return JsonResponse({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Error getting model status: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def process_frame_with_recognition(request):
    """
    Enhanced frame processing with face recognition for attendance
    Integrates with existing 30 FPS system
    """
    try:
        # Get the frame data from request
        data = json.loads(request.body)
        image_data = data.get('image')
        frame_number = data.get('frame_number', 0)
        session_id = data.get('session_id', None)  # For attendance session tracking
        
        if not image_data:
            return JsonResponse({'error': 'No image data provided'}, status=400)
        
        # Decode the base64 image
        try:
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return JsonResponse({'error': 'Invalid image data'}, status=400)
        
        # Step 1 & 2: Face detection using existing service
        faces_detected, face_data = face_detection_service.detect_faces(frame)
        
        response_data = {
            'frame_number': frame_number,
            'timestamp': timezone.now().isoformat(),
            'face_detected': faces_detected,
            'fps_info': {
                'target_fps': 30,
                'frame_interval_ms': 33
            }
        }
        
        if faces_detected and face_data:
            # Step 3: Preprocessing using existing service
            gray_frame = face_detection_service.preprocess_frame_30fps(frame)
            
            # Step 4 & 5: Face Recognition (NEW)
            student_id, confidence, student_name = face_recognition_service.recognize_face(frame)
            
            response_data.update({
                'preprocessing_applied': True,
                'recognition_attempted': True,
                'student_recognized': student_id is not None,
                'student_id': student_id,
                'student_name': student_name,
                'confidence': confidence,
                'quality_score': face_data.get('quality_score', 0)
            })
            
            # Step 6: Attendance Marking (if confidence is high enough)
            attendance_marked = False
            if student_id and confidence >= 0.8:  # 80% confidence threshold
                try:
                    student = Student.objects.get(student_id=student_id)
                    today = timezone.now().date()
                    
                    # Get or create attendance record for today
                    attendance_record, created = AttendanceRecord.objects.get_or_create(
                        student=student,
                        date=today,
                        defaults={
                            'recognition_confidence': confidence,
                            'recognition_method': 'face_recognition'
                        }
                    )
                    
                    # Mark entry if not already marked or if this is better confidence
                    if created or not attendance_record.entry_time or confidence > attendance_record.recognition_confidence:
                        attendance_record.mark_entry(confidence)
                        attendance_marked = True
                        
                        logger.info(f"Attendance marked for {student_name} (ID: {student_id}) with confidence {confidence:.2f}")
                    
                    response_data['attendance_marked'] = attendance_marked
                    response_data['attendance_record_id'] = attendance_record.id
                    
                except Student.DoesNotExist:
                    logger.warning(f"Student {student_id} not found in database")
                except Exception as e:
                    logger.error(f"Error marking attendance: {str(e)}")
            
            # Log recognition attempt for analytics
            if session_id:
                try:
                    session = AttendanceSession.objects.get(id=session_id, is_active=True)
                    RecognitionLog.objects.create(
                        attendance_session=session,
                        student=Student.objects.get(student_id=student_id) if student_id else None,
                        confidence_score=confidence,
                        recognition_successful=student_id is not None,
                        attendance_marked=attendance_marked,
                        image_quality_score=face_data.get('quality_score', 0),
                        face_size=f"{face_data.get('width', 0)}x{face_data.get('height', 0)}"
                    )
                except (AttendanceSession.DoesNotExist, Student.DoesNotExist):
                    pass  # Session tracking is optional
        
        else:
            # No face detected
            response_data.update({
                'preprocessing_applied': False,
                'recognition_attempted': False,
                'student_recognized': False,
                'message': 'No face detected in frame'
            })
        
        # Convert numpy types for JSON serialization
        response_data = make_json_serializable(response_data)
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error in process_frame_with_recognition: {str(e)}")
        return JsonResponse({
            'error': 'Internal server error',
            'frame_number': frame_number,
            'timestamp': timezone.now().isoformat()
        }, status=500)


def attendance_dashboard(request):
    """
    Dashboard for viewing attendance records and managing attendance sessions
    """
    # Get attendance statistics
    today = timezone.now().date()
    
    # Recent attendance records
    recent_attendance = AttendanceRecord.objects.filter(
        date=today
    ).select_related('student').order_by('-entry_time')
    
    # Attendance statistics
    total_students = Student.objects.filter(is_active=True).count()
    present_today = recent_attendance.filter(status='present').count()
    absent_today = total_students - present_today
    
    # Active attendance session
    active_session = AttendanceSession.objects.filter(is_active=True).first()
    
    # Recognition model status
    model_stats = face_recognition_service.get_model_statistics()
    
    context = {
        'recent_attendance': recent_attendance,
        'total_students': total_students,
        'present_today': present_today,
        'absent_today': absent_today,
        'attendance_percentage': (present_today / total_students * 100) if total_students > 0 else 0,
        'active_session': active_session,
        'model_stats': model_stats,
        'today': today
    }
    
    return render(request, 'main_app/attendance_dashboard.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def start_attendance_session(request):
    """
    Start a new attendance monitoring session
    """
    try:
        data = json.loads(request.body)
        class_name = data.get('class_name', 'Default Class')
        recognition_threshold = data.get('recognition_threshold', 0.8)
        
        # End any existing active sessions
        AttendanceSession.objects.filter(is_active=True).update(is_active=False)
        
        # Create new session
        session = AttendanceSession.objects.create(
            class_name=class_name,
            start_time=timezone.now().time(),
            recognition_threshold=recognition_threshold,
            is_active=True
        )
        
        logger.info(f"Started new attendance session: {class_name}")
        
        return JsonResponse({
            'success': True,
            'session_id': session.id,
            'message': f'Attendance session started for {class_name}'
        })
        
    except Exception as e:
        logger.error(f"Error starting attendance session: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Error starting session: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def end_attendance_session(request, session_id):
    """
    End an active attendance session
    """
    try:
        session = get_object_or_404(AttendanceSession, id=session_id, is_active=True)
        
        session.end_time = timezone.now().time()
        session.is_active = False
        
        # Calculate session statistics
        recognition_logs = session.recognition_logs.all()
        session.total_recognitions = recognition_logs.count()
        session.unique_students_detected = recognition_logs.filter(
            recognition_successful=True
        ).values('student').distinct().count()
        
        session.save()
        
        logger.info(f"Ended attendance session {session.class_name}")
        
        return JsonResponse({
            'success': True,
            'message': f'Attendance session ended',
            'statistics': {
                'total_recognitions': session.total_recognitions,
                'unique_students': session.unique_students_detected,
                'duration': str(session.duration) if session.duration else None
            }
        })
        
    except Exception as e:
        logger.error(f"Error ending attendance session: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Error ending session: {str(e)}'
        })
