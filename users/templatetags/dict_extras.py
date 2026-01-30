# -*- coding: utf-8 -*-
from django import template

register = template.Library()

@register.filter
def get_item(d, key):
    """
    Safe dict getter for Django templates.

    Usage:
      {{ mydict|get_item:bucket }}
      {% if mydict|get_item:bucket != None %} ... {% endif %}
    """
    if d is None:
        return None
    try:
        return d.get(key)
    except Exception:
        try:
            return d[key]
        except Exception:
            return None
