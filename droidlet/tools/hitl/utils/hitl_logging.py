import logging
import os

"""
Logger need:
- Log file saved location 
    - S3_BUCKET_NAME
    - S3_ROOT
- Class name
- Format

Logging - same as python logging
"""

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)

class HitlLogger():
    def __init__(self, logger_name: str, batch_id: int, formatter=log_formatter, level=logging.WARNING):
        log_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/pipeline_logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = f"{log_dir}/{logger_name}.log"
        fh = logging.FileHandler(log_file)        
        fh.setFormatter(formatter)

        sh = logging.StreamHandler() 
        sh.setFormatter(formatter)

        logger = logging.getLogger(logger_name)

        logger.setLevel(level)
        logger.addHandler(fh)
        logger.addHandler(sh)

        self.__logger__ = logger
    
    def get_logger(self):
        return self.__logger__


