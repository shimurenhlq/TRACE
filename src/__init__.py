"""
TRACE: Three-stage Agentic RAG for Complex Multi-modal Document QA
"""

from .config import Config, ModelConfig
from .environment import BookEnvironment
from .agents import AgenticSystem
from .prompts import PLANNER_SYSTEM_PROMPT, NAVIGATOR_VLM_PROMPT_TEMPLATE, REASONER_SYSTEM_PROMPT

__version__ = "0.1.0"
__all__ = [
    "Config",
    "ModelConfig",
    "BookEnvironment",
    "AgenticSystem",
    "PLANNER_SYSTEM_PROMPT",
    "NAVIGATOR_VLM_PROMPT_TEMPLATE",
    "REASONER_SYSTEM_PROMPT",
]
