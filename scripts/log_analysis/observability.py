import logging

logger = logging.getLogger(__name__)


def init_observability() -> None:
    logger.info("Observability disabled; using local logger output only.")
