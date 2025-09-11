from django.contrib import admin
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.urls import path
from django.shortcuts import render, redirect
from django.utils.html import format_html
from .models import Student, EnrollmentImage, EnrollmentSession, SystemConfiguration
from .config_service import config_service
import json


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['student_id', 'full_name', 'enrollment_date', 'total_images_captured', 'enrollment_completed', 'is_active', 'images_count_display']
    list_filter = ['enrollment_completed', 'is_active', 'enrollment_date']
    search_fields = ['student_id', 'full_name', 'email']
    readonly_fields = ['enrollment_date', 'total_images_captured']
    actions = ['delete_selected_with_files', 'cleanup_student_files']
    
    def images_count_display(self, obj):
        """Display the number of enrollment images"""
        count = obj.enrollment_images.count()
        return format_html(
            '<span style="color: {};">{} images</span>',
            'green' if count > 0 else 'gray',
            count
        )
    images_count_display.short_description = 'Images'
    
    def delete_selected_with_files(self, request, queryset):
        """Custom delete action with file cleanup warning"""
        if request.POST.get('post'):
            # User confirmed deletion
            total_students = queryset.count()
            total_images = sum(student.enrollment_images.count() for student in queryset)
            
            # Delete students (this will trigger our cleanup)
            for student in queryset:
                student.delete()
            
            self.message_user(
                request,
                f'Successfully deleted {total_students} students and {total_images} associated image files.',
                messages.SUCCESS
            )
        else:
            # Show confirmation page with file counts
            context = {
                'title': 'Delete Students and Associated Files',
                'students': queryset,
                'total_students': queryset.count(),
                'total_images': sum(student.enrollment_images.count() for student in queryset),
                'action_checkbox_name': admin.ACTION_CHECKBOX_NAME,
            }
            return render(request, 'admin/delete_students_confirmation.html', context)
    
    delete_selected_with_files.short_description = "Delete selected students and all their files"
    
    def cleanup_student_files(self, request, queryset):
        """Clean up orphaned files for selected students"""
        from .signals import cleanup_orphaned_files
        
        cleaned_count = 0
        for student in queryset:
            try:
                cleanup_orphaned_files(student.student_id)
                cleaned_count += 1
            except Exception as e:
                self.message_user(
                    request,
                    f'Error cleaning files for {student.student_id}: {str(e)}',
                    messages.ERROR
                )
        
        self.message_user(
            request,
            f'File cleanup completed for {cleaned_count} students.',
            messages.SUCCESS
        )
    
    cleanup_student_files.short_description = "Clean up orphaned files for selected students"
    
    def delete_model(self, request, obj):
        """Override delete to show warning message"""
        image_count = obj.enrollment_images.count()
        super().delete_model(request, obj)
        
        self.message_user(
            request,
            f'Student {obj.student_id} deleted along with {image_count} image files.',
            messages.SUCCESS
        )
    
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


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'current_value_display', 'setting_type', 'is_active', 'requires_restart', 'last_modified']
    list_filter = ['category', 'setting_type', 'is_active', 'requires_restart']
    search_fields = ['name', 'key', 'description']
    readonly_fields = ['last_modified', 'last_modified_by']
    
    fieldsets = (
        ('Setting Information', {
            'fields': ('key', 'name', 'description', 'category')
        }),
        ('Value Configuration', {
            'fields': ('setting_type', 'value', 'default_value')
        }),
        ('Validation', {
            'fields': ('min_value', 'max_value', 'validation_regex'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('is_active', 'requires_restart', 'last_modified', 'last_modified_by'),
            'classes': ('collapse',)
        }),
    )
    
    def current_value_display(self, obj):
        """Display the current parsed value"""
        try:
            value = obj.get_value()
            if obj.setting_type == 'JSON':
                return format_html('<code>{}</code>', json.dumps(value, indent=2))
            elif obj.setting_type == 'BOOLEAN':
                return format_html(
                    '<span style="color: {};">{}</span>',
                    'green' if value else 'red',
                    '✓ True' if value else '✗ False'
                )
            else:
                return str(value)
        except Exception as e:
            return format_html('<span style="color: red;">Error: {}</span>', str(e))
    
    current_value_display.short_description = 'Current Value'
    current_value_display.admin_order_field = 'value'
    
    def save_model(self, request, obj, form, change):
        """Save model with user tracking"""
        if request.user.is_authenticated:
            obj.last_modified_by = request.user.username
        super().save_model(request, obj, form, change)
        
        # Clear cache when settings change
        config_service.clear_cache()
        
        if obj.requires_restart:
            messages.warning(
                request,
                f'Setting "{obj.name}" requires a server restart to take effect.'
            )
    
    def get_urls(self):
        """Add custom admin URLs"""
        urls = super().get_urls()
        custom_urls = [
            path('export/', self.admin_site.admin_view(self.export_settings), name='export_settings'),
            path('import/', self.admin_site.admin_view(self.import_settings), name='import_settings'),
            path('reset/', self.admin_site.admin_view(self.reset_settings), name='reset_settings'),
            path('preview/', self.admin_site.admin_view(self.preview_settings), name='preview_settings'),
        ]
        return custom_urls + urls
    
    def export_settings(self, request):
        """Export all settings as JSON"""
        try:
            settings_data = config_service.export_settings()
            response = HttpResponse(
                json.dumps(settings_data, indent=2),
                content_type='application/json'
            )
            response['Content-Disposition'] = 'attachment; filename="system_settings.json"'
            return response
        except Exception as e:
            messages.error(request, f'Export failed: {str(e)}')
            return redirect('admin:main_app_systemconfiguration_changelist')
    
    def import_settings(self, request):
        """Import settings from JSON file"""
        if request.method == 'POST' and request.FILES.get('settings_file'):
            try:
                settings_file = request.FILES['settings_file']
                settings_data = json.loads(settings_file.read().decode('utf-8'))
                
                imported_count = config_service.import_settings(
                    settings_data,
                    modified_by=request.user.username if request.user.is_authenticated else 'admin'
                )
                
                messages.success(request, f'Successfully imported {imported_count} settings.')
                
            except Exception as e:
                messages.error(request, f'Import failed: {str(e)}')
            
            return redirect('admin:main_app_systemconfiguration_changelist')
        
        return render(request, 'admin/import_settings.html')
    
    def reset_settings(self, request):
        """Reset all settings to default values"""
        if request.method == 'POST':
            try:
                reset_count = 0
                for config in SystemConfiguration.objects.all():
                    config.value = config.default_value
                    config.last_modified_by = request.user.username if request.user.is_authenticated else 'admin'
                    config.save()
                    reset_count += 1
                
                config_service.clear_cache()
                messages.success(request, f'Successfully reset {reset_count} settings to default values.')
                
            except Exception as e:
                messages.error(request, f'Reset failed: {str(e)}')
            
            return redirect('admin:main_app_systemconfiguration_changelist')
        
        return render(request, 'admin/reset_settings.html')
    
    def preview_settings(self, request):
        """Preview current settings grouped by category"""
        try:
            all_settings = config_service.get_all_settings()
            context = {
                'title': 'System Settings Preview',
                'settings': all_settings,
            }
            return render(request, 'admin/preview_settings.html', context)
            
        except Exception as e:
            messages.error(request, f'Preview failed: {str(e)}')
            return redirect('admin:main_app_systemconfiguration_changelist')
