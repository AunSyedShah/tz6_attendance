#!/usr/bin/env python
"""
Django Setup Script for AI Attendance System
Converts Streamlit app to Django web application
"""

import os
import sys
import django
from django.core.management import execute_from_command_line

def setup_django():
    """Set up Django environment and run initial setup commands"""
    
    # Set Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tz6_attendance.settings')
    
    # Initialize Django
    django.setup()
    
    print("🚀 Setting up AI Attendance System (Django)")
    print("=" * 50)
    
    # Commands to run
    commands = [
        ['makemigrations'],
        ['migrate'],
        ['collectstatic', '--noinput'],
    ]
    
    for command in commands:
        print(f"\n📋 Running: python manage.py {' '.join(command)}")
        try:
            execute_from_command_line(['manage.py'] + command)
            print(f"✅ Command completed successfully")
        except Exception as e:
            print(f"❌ Error running command: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("🎉 Django setup completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Create superuser: python manage.py createsuperuser")
    print("3. Run development server: python manage.py runserver")
    print("4. Access the application at: http://127.0.0.1:8000/")
    print("\n📚 Features converted from Streamlit:")
    print("- Student enrollment with camera capture")
    print("- Face detection and validation")
    print("- Real-time progress tracking")
    print("- Demo mode for file uploads")
    print("- Interactive prompts for comprehensive face capture")
    print("- Django admin interface for data management")

if __name__ == '__main__':
    setup_django()
