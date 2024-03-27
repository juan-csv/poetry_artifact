"""
Logger class to log the output of the program in each file
"""
import glob
import logging
import os
import sys
import time


class StringFormatter(logging.Formatter):
    """
    String formatter
    """
    def format(self, record):
        record.msg = str(record.msg)  # convert the log message to a string
        return super().format(record)


class Logger:
    """
    Logger
    """
    def __init__(self, name, level=logging.DEBUG):
        self.name = name
        self.level = level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.formatter = StringFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self._add_console_handler()
        # self._add_file_handler()

    def _add_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        logs_folder = 'logs'
        if not os.path.exists(logs_folder):
            os.mkdir(logs_folder)
        log_file_path = os.path.join(logs_folder, 'info.log')
        self._delete_old_logs(logs_folder)
        file_handler = logging.FileHandler(log_file_path, mode='a')  # overwrite the file
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    @staticmethod
    def _delete_old_logs(logs_folder):
        # Delete log file if it exists
        log_file_path = os.path.join(logs_folder, 'info.log')
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

        # Delete old log files (older than 1 week)
        current_time = time.time()
        for log_file in glob.glob(os.path.join(logs_folder, '.log')):
            if os.stat(log_file).st_mtime < current_time - 7 * 24 * 60 * 60:
                os.remove(log_file)

    def get_logger(self):
        """
        Get logger
        """
        return self.logger

    def get_formatter(self):
        """
        Get formatter
        """
        return self.formatter
