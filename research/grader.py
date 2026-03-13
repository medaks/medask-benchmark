"""
LLM-as-judge grader for triage benchmark responses.

Instead of relying on regex extraction to match em/ne/sc tokens,
this uses a second LLM to evaluate whether the model's raw response
indicates the correct triage urgency level.
"""
import re
from typing import Optional, Tuple

from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.ummon.lmstudio import UmmonLMStudio


_GRADER_PROMPT = """You are a medical triage evaluation judge. Your task is to determine what triage urgency level a model's response indicates.

The three triage levels are:
- **em** (Emergency): Conditions requiring immediate emergency care
- **ne** (Non-Emergency): Conditions requiring medical attention but not emergency
- **sc** (Self-Care): Conditions that can be managed at home

## Clinical Vignette
{vignette}

## Model's Response
{response}

## Instructions
Based on the model's response above, determine which single triage level (em, ne, or sc) the model is recommending.

Consider the full meaning of the response, not just keyword matching. For example:
- "This requires immediate emergency treatment" → em
- "See a doctor soon but this is not an emergency" → ne
- "This can be managed at home with rest" → sc

Respond with ONLY one of: em, ne, sc"""


def build_grader_client(
    model: str,
    base_url: str = "http://localhost:1234/v1",
) -> UmmonLMStudio:
    """Build an LMStudio client configured for grading (low temp, short output)."""
    return UmmonLMStudio(
        model=model,
        temperature=0.1,
        max_tokens=10,
        base_url=base_url,
    )


def grade_response(
    grader_client: UmmonLMStudio,
    raw_response: str,
    vignette_text: str,
) -> str:
    """Use an LLM grader to extract the triage level from a raw model response.

    Returns one of: 'em', 'ne', 'sc', or the raw grader output (truncated)
    if parsing fails.
    """
    prompt_text = _GRADER_PROMPT.format(
        vignette=vignette_text,
        response=raw_response,
    )

    grader_raw = grader_client.inquire(
        CMessage(user_id=1, body=prompt_text, role=Role.USER)
    ).body

    cleaned = re.sub(r"[`\s]", " ", grader_raw.lower()).strip()
    match = re.search(r"\b(em|ne|sc)\b", cleaned)
    return match.group(1) if match else cleaned[:50]
