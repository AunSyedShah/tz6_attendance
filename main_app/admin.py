from django.contrib import admin
from .models import Student, EnrollmentImage, EnrollmentSession


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['student_id', 'full_name', 'enrollment_date', 'total_images_captured', 'enrollment_completed', 'is_active']
    list_filter = ['enrollment_completed', 'is_active', 'enrollment_date']
    search_fields = ['student_id', 'full_name', 'email']
    readonly_fields = ['enrollment_date', 'total_images_captured']
    
    fieldsets = (
        ('Student Information', {
            'fields': ('student_id', 'full_name', 'email', 'phone')
        }),
        ('Enrollment Status', {
            'fields': ('enrollment_completed', 'total_images_captured', 'is_active')
        }),
        ('Timestamps', {
            'fields': ('enrollment_date',),
            'classes': ('collapse',)
        }),
    )


@admin.register(EnrollmentImage)
class EnrollmentImageAdmin(admin.ModelAdmin):
    list_display = ['student', 'prompt_index', 'image_sequence', 'confidence_score', 'capture_mode', 'capture_timestamp']
    list_filter = ['capture_mode', 'prompt_index', 'capture_timestamp']
    search_fields = ['student__student_id', 'student__full_name']
    readonly_fields = ['capture_timestamp', 'face_coordinates']
    
    fieldsets = (
        ('Image Information', {
            'fields': ('student', 'image', 'prompt_index', 'image_sequence')
        }),
        ('Quality Metrics', {
            'fields': ('confidence_score', 'brightness_score', 'blur_score', 'face_coordinates')
        }),
        ('Capture Details', {
            'fields': ('capture_mode', 'capture_timestamp')
        }),
    )


@admin.register(EnrollmentSession)
class EnrollmentSessionAdmin(admin.ModelAdmin):
    list_display = ['student', 'session_mode', 'session_start', 'session_completed', 'successful_captures', 'success_rate_display']
    list_filter = ['session_mode', 'session_completed', 'session_start']
    search_fields = ['student__student_id', 'student__full_name']
    readonly_fields = ['session_start', 'success_rate_display', 'duration_display']
    
    def success_rate_display(self, obj):
        return f"{obj.success_rate:.1f}%" if obj.total_attempts > 0 else "N/A"
    success_rate_display.short_description = 'Success Rate'
    
    def duration_display(self, obj):
        return str(obj.duration) if obj.session_end else "In Progress"
    duration_display.short_description = 'Duration'
    
    fieldsets = (
        ('Session Information', {
            'fields': ('student', 'session_mode', 'session_start', 'session_end', 'session_completed')
        }),
        ('Statistics', {
            'fields': ('total_attempts', 'successful_captures', 'failed_captures', 'success_rate_display', 'duration_display')
        }),
        ('Technical Details', {
            'fields': ('user_agent', 'ip_address'),
            'classes': ('collapse',)
        }),
    )
