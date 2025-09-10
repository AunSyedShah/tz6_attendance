"""
Custom template filters for the AI Attendance System
"""
from django import template

register = template.Library()


@register.filter
def percentage(value, total):
    """
    Calculate percentage given a value and total
    Usage: {{ captured_images|percentage:total_images }}
    """
    try:
        if total == 0:
            return 0
        return min(100, (value / total) * 100)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


@register.filter
def divide(value, divisor):
    """
    Divide two numbers
    Usage: {{ number|divide:10 }}
    """
    try:
        if divisor == 0:
            return 0
        return value / divisor
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


@register.filter
def multiply(value, multiplier):
    """
    Multiply two numbers
    Usage: {{ number|multiply:10 }}
    """
    try:
        return value * multiplier
    except (ValueError, TypeError):
        return 0


@register.filter
def subtract(value, subtractor):
    """
    Subtract two numbers
    Usage: {{ number|subtract:10 }}
    """
    try:
        return value - subtractor
    except (ValueError, TypeError):
        return 0


@register.filter
def add_number(value, addend):
    """
    Add two numbers (Django has built-in 'add' but this is more explicit)
    Usage: {{ number|add_number:10 }}
    """
    try:
        return value + addend
    except (ValueError, TypeError):
        return 0


@register.simple_tag
def progress_bar_width(captured, total):
    """
    Calculate progress bar width percentage
    Usage: {% progress_bar_width captured_images total_images %}
    """
    try:
        if total == 0:
            return 0
        return min(100, (captured / total) * 100)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


@register.filter
def stroke_dashoffset(progress_percentage):
    """
    Calculate stroke-dashoffset for progress circle
    Usage: {{ progress_percentage|stroke_dashoffset }}
    """
    try:
        # For a circle with radius 42, circumference is 2 * π * 42 ≈ 264.16
        circumference = 264.16
        # Calculate offset based on progress (inverted for clockwise progress)
        offset = circumference - (progress_percentage / 100) * circumference
        return offset
    except (ValueError, TypeError):
        return 264.16


@register.filter
def lookup(dictionary, key):
    """
    Lookup a value in a dictionary by key
    Usage: {{ dict|lookup:key }}
    """
    try:
        return dictionary.get(key)
    except (AttributeError, TypeError):
        return None
