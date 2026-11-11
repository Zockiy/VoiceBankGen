"""Built-in language pack registration."""

from .base import language, profile, register_language, register_profile
from .ja import JapaneseCVProfile, JapanesePack
from .ja_cvvc import JapaneseCVVCProfile
from .zh import MandarinPack, MandarinCVProfile, MandarinCVVCProfile

register_language(JapanesePack())
register_profile(JapaneseCVProfile())
register_profile(JapaneseCVVCProfile())
register_language(MandarinPack())
register_profile(MandarinCVProfile())
register_profile(MandarinCVVCProfile())

__all__ = ["language", "profile", "register_language", "register_profile"]
