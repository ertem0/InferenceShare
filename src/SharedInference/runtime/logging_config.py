"""Console logging configured explicitly by the application entry point."""

import logging
from copy import copy


class ConsoleFormatter(logging.Formatter):
    """Shorten the displayed component without changing the original record."""

    def format(self, record: logging.LogRecord) -> str:
        display_record = copy(record)
        display_record.name = record.name.rsplit(".", 1)[-1]
        return super().format(display_record)


def configure_logging(level: str = "INFO") -> None:
    """Configure application logs without changing the root logger's handlers."""
    application = logging.getLogger("SharedInference")
    handler = logging.StreamHandler()
    handler.setFormatter(
        ConsoleFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
            datefmt="%H:%M:%S" if level == "DEBUG" else "%Y-%m-%d %H:%M:%S",
        )
    )
    for existing in application.handlers[:]:
        application.removeHandler(existing)
        existing.close()
    application.addHandler(handler)
    application.setLevel(level)
    application.propagate = False
