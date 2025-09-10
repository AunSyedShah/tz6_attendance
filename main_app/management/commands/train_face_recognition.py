"""
Django management command to train the face recognition model
Usage: python manage.py train_face_recognition
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from main_app.face_recognition_service import face_recognition_service
from main_app.models import Student

class Command(BaseCommand):
    help = 'Train the face recognition model using enrolled student data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force retrain even if model already exists'
        )
        parser.add_argument(
            '--student-id',
            type=str,
            help='Train for specific student only'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting Face Recognition Model Training...')
        )
        
        # Check enrolled students
        enrolled_students = Student.objects.filter(enrollment_completed=True)
        if not enrolled_students.exists():
            self.stdout.write(
                self.style.ERROR('No enrolled students found. Please complete enrollment first.')
            )
            return
        
        self.stdout.write(f'Found {enrolled_students.count()} enrolled students:')
        for student in enrolled_students:
            image_count = student.enrollment_images.count()
            self.stdout.write(f'  - {student.student_id}: {student.full_name} ({image_count} images)')
        
        # Train specific student or all students
        if options['student_id']:
            student_id = options['student_id']
            try:
                student = Student.objects.get(student_id=student_id)
                self.stdout.write(f'\nTraining model for student: {student.full_name}')
                
                encoding_count = face_recognition_service.generate_face_encodings(student_id)
                if encoding_count > 0:
                    success = face_recognition_service.retrain_for_student(student_id)
                    if success:
                        self.stdout.write(
                            self.style.SUCCESS(f'Successfully trained model for {student.full_name}')
                        )
                    else:
                        self.stdout.write(
                            self.style.ERROR(f'Failed to train model for {student.full_name}')
                        )
                else:
                    self.stdout.write(
                        self.style.ERROR(f'No valid face encodings generated for {student.full_name}')
                    )
                    
            except Student.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'Student with ID {student_id} not found')
                )
                return
        else:
            # Train full model
            self.stdout.write('\nTraining full face recognition model...')
            
            start_time = timezone.now()
            success = face_recognition_service.train_recognition_model()
            end_time = timezone.now()
            
            if success:
                # Get model statistics
                stats = face_recognition_service.get_model_statistics()
                duration = (end_time - start_time).total_seconds()
                
                self.stdout.write(
                    self.style.SUCCESS(f'\n✅ Model training completed successfully!')
                )
                self.stdout.write(f'Training time: {duration:.2f} seconds')
                self.stdout.write(f'Total students: {stats["total_students"]}')
                self.stdout.write(f'Total face encodings: {stats["total_encodings"]}')
                
                # Show per-student breakdown
                self.stdout.write('\nPer-student breakdown:')
                for student_key, info in stats['students_mapping'].items():
                    self.stdout.write(f'  - {info["name"]} (ID: {info["student_id"]}): {info["encoding_count"]} encodings')
                
            else:
                self.stdout.write(
                    self.style.ERROR('❌ Model training failed. Check logs for details.')
                )
        
        # Test model if training was successful
        if face_recognition_service.classifier is not None:
            self.stdout.write('\n📊 Model Status:')
            stats = face_recognition_service.get_model_statistics()
            
            if stats['model_trained']:
                self.stdout.write(
                    self.style.SUCCESS('✅ Face recognition model is ready for attendance tracking!')
                )
                self.stdout.write('\nNext steps:')
                self.stdout.write('1. Start the Django server: python manage.py runserver')
                self.stdout.write('2. Go to /attendance/ to start an attendance session')
                self.stdout.write('3. Use the 30 FPS camera system with recognition_mode=true')
            else:
                self.stdout.write(
                    self.style.WARNING('⚠️  Model training completed but may need verification')
                )
        else:
            self.stdout.write(
                self.style.ERROR('❌ No trained model available')
            )
