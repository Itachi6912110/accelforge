"""
AccelForge custom hwcomponents models.

Auto-discovered by hwc.get_models() because the package name starts with
'hwcomponents_'. Add new ComponentModel subclasses here or import them from
submodules to make them available project-wide.
"""

from hwcomponents_extended.cache import SimpleSRAMCache

__all__ = ["SimpleSRAMCache"]
