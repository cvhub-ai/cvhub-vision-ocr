import logging
import sys


class LoggingConfig:
    """Configure application-wide logging."""

    LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def configure(log_level: str = "INFO") -> None:
        level = getattr(
            logging,
            log_level.upper(),
            logging.INFO,
        )

        logging.basicConfig(
            level=level,
            format=LoggingConfig.LOG_FORMAT,
            datefmt=LoggingConfig.DATE_FORMAT,
            handlers=[
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
