"""
Configuration status view for admin dashboard
"""
from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
from django.http import JsonResponse
from .config_service import config_service
from .models import SystemConfiguration
import json


@staff_member_required
def configuration_status(request):
    """Display configuration status and controls"""
    try:
        all_settings = config_service.get_all_settings()
        
        # Get configuration statistics
        total_settings = SystemConfiguration.objects.count()
        active_settings = SystemConfiguration.objects.filter(is_active=True).count()
        categories = SystemConfiguration.objects.values_list('category', flat=True).distinct()
        
        context = {
            'title': 'System Configuration Status',
            'all_settings': all_settings,
            'total_settings': total_settings,
            'active_settings': active_settings,
            'categories': sorted(categories),
            'has_dynamic_config': True,
        }
        
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse(context)
        
        return render(request, 'admin/configuration_status.html', context)
        
    except Exception as e:
        context = {
            'title': 'Configuration Status - Error',
            'error': str(e),
            'has_dynamic_config': False,
        }
        
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse(context, status=500)
        
        return render(request, 'admin/configuration_status.html', context)


def api_reload_configuration(request):
    """API endpoint to reload configuration"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    try:
        # Reload configuration in services
        from .services import face_detection_service
        face_detection_service.reload_configuration()
        
        # Clear configuration cache
        config_service.clear_cache()
        
        return JsonResponse({
            'success': True,
            'message': 'Configuration reloaded successfully'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
