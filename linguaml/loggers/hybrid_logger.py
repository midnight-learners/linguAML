from .common import logger
from linguaml.types import TunerRole

hybrid_logger = logger.bind(tuner_role=TunerRole.HYBRID)
