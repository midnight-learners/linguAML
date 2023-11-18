from .common import logger
from linguaml.types import TunerRole

llm_logger = logger.bind(tuner_role=TunerRole.LLM)
