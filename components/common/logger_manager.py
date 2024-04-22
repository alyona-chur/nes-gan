"""This module contains methods and classes for logging."""
from abc import ABC
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import sys
from typing import Optional


from components.common.constants import LOGGER_NAME


def set_logging(log_file_path: Optional[Path] = None):
    """Configures writing logs.

    Args:
        log_file_path: A path to log file if writing to file is required.
    """
    if log_file_path is not None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    level = logging.INFO

    # Create logger
    logging.basicConfig(level=level,
                        stream=sys.stdout,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')

    # Add file writer handler if needed
    if log_file_path is not None:
        file_handler = TimedRotatingFileHandler(log_file_path,
                                                when='midnight',
                                                interval=1,
                                                backupCount=1000)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'))
        file_handler.setLevel(level)
        logging.getLogger('').addHandler(file_handler)


# def get_logger_name(class_name: str) -> str:
#     """Returns logger name based on a class name.

#     Args:
#         class_name: Class name.

#     Returns:
#         Logger name.
#     """
#     return f'[{class_name}]'


class LoggerManager:
    """A class to manage logging.

    Attributes:
        _local_sender: Str, local logger name.
        _cur_sender: Str, the current logger name.
        _logger: logging.Logger, the current logger instance.
    """
    def __init__(self, log_file_path: Optional[Path] = None):
        """Initializes an instance of the class."""
        set_logging(log_file_path)
        self._local_sender = LOGGER_NAME.format(class_name=self.__class__.__name__)
        self._cur_sender = self._local_sender
        self._logger = logging.getLogger(self._cur_sender)

    def log(self,
            logger_name: str,
            log_level: 'logging.LEVEL',
            message: str):
        """Logs a message."""
        if logger_name != self._cur_sender:
            self._cur_sender = logger_name
            self._logger = logging.getLogger(self._cur_sender)

        if log_level == logging.INFO:
            self._logger.info(message)
        elif log_level == logging.WARNING:
            self._logger.warning(message)
        elif log_level == logging.ERROR:
            self._logger.error(message)
        elif log_level == logging.DEBUG:
            self._logger.debug(message)
        else:
            self._logger = logging.getLogger(self._local_sender)
            self._logger.error(f'Unknown log level: {log_level} for {message}.')


class LogWritingClass(ABC):
    def __init__(self, logger_name: str, logger_manager: LoggerManager):
        self._logger_name = logger_name
        self._logger_manager = logger_manager

    def _log(self, log_level: 'logging.LEVEL', message: str):
        self._logger_manager.log(self._logger_name, log_level, message)
