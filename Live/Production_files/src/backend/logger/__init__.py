# logger/__init__.py
#
# Student hint: this is the "one place" from customer_logger.py's TODO #1.
# Every other file in the project does `from src.backend.logger import
# GLOBAL_LOGGER as log` — they all share this single instance.
from .customer_logger import CustomLogger


# Create a single shared logger instance
GLOBAL_LOGGER = CustomLogger().get_logger("multi_document_assist")
