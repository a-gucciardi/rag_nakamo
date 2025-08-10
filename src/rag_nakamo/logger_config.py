import logging

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{self.RESET}"
        record.name = f"\033[94m{record.name}{self.RESET}"  # Blue
        return super().format(record)

def setup_logging():
    formatter = ColorFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)