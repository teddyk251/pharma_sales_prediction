import logging.handlers as loghandler
import logging
import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


loggers = {}
LOG_LEVEL = logging.DEBUG


def setup_custom_logger(name: str, file_name=None, log_level=LOG_LEVEL):
    '''
    Setup a custom logger with a given name and log level.
    Args:
        name: The name of the logger.
        file_name: The name of the log file.
        log_level: The log level.
    '''

    file = False
    if file_name:
        file = True

    if loggers.get(name.lower()):
        return loggers[name.lower()]

    logger = logging.getLogger(name.lower())
    loggers[name] = logger

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - line %(lineno)d - %(message)s')
    if file:
        handler = loghandler.TimedRotatingFileHandler(
            filename=file_name,
            when='D', interval=1, backupCount=7,
            encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = True
    return logger
