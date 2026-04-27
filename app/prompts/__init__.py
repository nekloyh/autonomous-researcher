"""Prompt registry."""
from app.prompts.critic import CRITIC_PROMPT
from app.prompts.critic import PROMPT_VERSION as CRITIC_VERSION
from app.prompts.planner import PLANNER_PROMPT
from app.prompts.planner import PROMPT_VERSION as PLANNER_VERSION
from app.prompts.researcher import PROMPT_VERSION as RESEARCHER_VERSION
from app.prompts.researcher import RESEARCHER_PROMPT
from app.prompts.synthesizer import PROMPT_VERSION as SYNTHESIZER_VERSION
from app.prompts.synthesizer import SYNTHESIZER_PROMPT

PROMPT_VERSIONS = {
    "planner": PLANNER_VERSION,
    "researcher": RESEARCHER_VERSION,
    "synthesizer": SYNTHESIZER_VERSION,
    "critic": CRITIC_VERSION,
}

__all__ = [
    "PLANNER_PROMPT",
    "RESEARCHER_PROMPT",
    "SYNTHESIZER_PROMPT",
    "CRITIC_PROMPT",
    "PROMPT_VERSIONS",
]
