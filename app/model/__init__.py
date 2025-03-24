# Import all models here for easy access
from app.model.user import User
from app.model.request import ProcessingRequest

# This ensures all models are imported when 'from app.model import *' is used
__all__ = ['User', 'ProcessingRequest']