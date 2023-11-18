from .common import logger
from linguaml.types import TunerRole

rl_logger = logger.bind(tuner_role=TunerRole.RL)
