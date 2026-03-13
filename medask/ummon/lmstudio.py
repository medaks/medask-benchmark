"""
LLM client for LMStudio local server.

LMStudio serves an OpenAI-compatible API at http://localhost:1234/v1.
The model name must match the model loaded in LMStudio.

Thinking models (qwen3.5, phi-4-reasoning, etc.) split their output
into reasoning_content and content.  When max_tokens is too low the
reasoning is truncated and content comes back empty.  This client
defaults to 2048 tokens and falls back to reasoning_content when
content is blank.
"""
from logging import getLogger
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.util.decorator import timeit
from medask.util.gen_cmsg import gen_cmsg
from medask.ummon.base import BaseUmmon

logger = getLogger("ummon.lmstudio")

_DEFAULT_BASE_URL = "http://localhost:1234/v1"


class UmmonLMStudio(BaseUmmon):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = OpenAI(
            api_key="lm-studio",
            base_url=base_url,
            timeout=120,
        )

    def _converse_raw(self, history: List[Dict[str, str]]) -> str:
        """Send a chat completion and return the useful text.

        For thinking models the answer lives in ``content`` only once the
        reasoning chain finishes.  If ``content`` is empty we fall back to
        ``reasoning_content`` so downstream parsing still has text to work
        with.
        """
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=history,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        msg = completion.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        # Thinking models with enough tokens: content has the clean answer.
        # Thinking models truncated:          content is empty, fall back.
        if not content.strip() and reasoning:
            content = reasoning

        return content

    def _converse_full(self, history: List[Dict[str, str]]) -> Tuple[str, str]:
        """Like _converse_raw but returns (content, reasoning_content)."""
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=history,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        msg = completion.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        if not content.strip() and reasoning:
            content = reasoning

        return content, reasoning

    @timeit(logger, log_kwargs=False)
    def inquire(self, prompt: CMessage, json: bool = False) -> CMessage:
        prompt_raw = prompt.to_openai()
        retort: str = self._converse_raw([prompt_raw])
        return gen_cmsg(prompt, body=retort, role=Role.ASSISTANT)

    @timeit(logger, log_kwargs=False)
    def converse(self, history: List[CMessage], json: bool = False) -> CMessage:
        history_raw = [msg.to_openai() for msg in history]
        retort: str = self._converse_raw(history_raw)
        return gen_cmsg(history[-1], body=retort, role=Role.ASSISTANT)
