from django.apps import AppConfig


class MainAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main_app'
    
    def ready(self):
        """Import signals when the app is ready"""
        import main_app.signals
