import logging
import re

import pytest

from SharedInference.runtime.logging_config import ConsoleFormatter, configure_logging


@pytest.mark.parametrize(
    ("level", "timestamp"),
    [
        ("DEBUG", r"\d{2}:\d{2}:\d{2}\.\d{3}"),
        ("INFO", r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}"),
    ],
)
def test_console_format_filtering_and_configuration_scope(
    monkeypatch, capsys, level, timestamp
):
    application = logging.getLogger("SharedInference")
    monkeypatch.setattr(application, "handlers", [])
    monkeypatch.setattr(application, "level", application.level)
    monkeypatch.setattr(application, "propagate", application.propagate)
    root_handlers = logging.getLogger().handlers[:]
    root_level = logging.getLogger().level
    configure_logging(level)
    configure_logging(level)  # Reconfiguration must not duplicate console output.
    try:
        for component in ("coordinator", "worker"):
            logger = logging.getLogger(f"SharedInference.runtime.{component}")
            logger.info("node=example ready experts=2")
            logger.debug("heartbeat sequence=1")
        lines = capsys.readouterr().err.splitlines()
        assert len(lines) == (4 if level == "DEBUG" else 2)
        assert all(
            re.match(
                timestamp + r" (INFO|DEBUG) (coordinator|worker) ",
                line,
            )
            for line in lines
        )
        assert logging.getLogger().handlers == root_handlers
        assert logging.getLogger().level == root_level
    finally:
        for handler in application.handlers:
            handler.close()
        application.handlers.clear()
        # Clear logging's level cache before monkeypatch restores the level.
        application.setLevel(logging.NOTSET)


def test_formatter_preserves_full_record_name():
    record = logging.LogRecord(
        "SharedInference.runtime.cli", logging.INFO, "", 0, "Started", (), None
    )
    assert ConsoleFormatter("%(name)s %(message)s").format(record) == "cli Started"
    assert record.name == "SharedInference.runtime.cli"
