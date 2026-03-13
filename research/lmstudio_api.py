"""
Utility to query LMStudio's API for available models and their metadata.

Uses the LMStudio-specific /api/v0/models endpoint which returns rich
metadata (arch, quantization, context length, state, capabilities).
Falls back to the OpenAI-compatible /v1/models for basic ID-only listing.
"""
from typing import Any, Dict, List, Optional

import requests


def _base_origin(base_url: str) -> str:
    """Strip /v1 suffix to get the server origin for LMStudio API calls."""
    return base_url.rstrip("/").removesuffix("/v1").removesuffix("/v0")


def fetch_models_detailed(
    base_url: str = "http://localhost:1234/v1",
) -> List[Dict[str, Any]]:
    """Fetch full model metadata from LMStudio's extended API.

    Returns a list of dicts with keys like:
        id, arch, quantization, publisher, compatibility_type,
        max_context_length, loaded_context_length, state, type, capabilities

    Sorted by state (loaded first) then id.
    Returns empty list on error.
    """
    origin = _base_origin(base_url)
    try:
        resp = requests.get(f"{origin}/api/v0/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        # Sort: loaded models first, then alphabetically
        models.sort(key=lambda m: (0 if m.get("state") == "loaded" else 1, m.get("id", "")))
        return models
    except Exception:
        return []


def fetch_models(base_url: str = "http://localhost:1234/v1") -> List[str]:
    """Fetch available model IDs from an LMStudio server.

    Tries the extended API first, falls back to OpenAI-compatible endpoint.
    Returns a sorted list of model ID strings, or an empty list on error.
    """
    detailed = fetch_models_detailed(base_url)
    if detailed:
        return [m["id"] for m in detailed]

    # Fallback to OpenAI-compatible endpoint
    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return sorted(m["id"] for m in data.get("data", []))
    except Exception:
        return []


def get_model_info(
    model_id: str,
    base_url: str = "http://localhost:1234/v1",
) -> Optional[Dict[str, Any]]:
    """Get metadata for a single model by ID.

    Returns the model dict or None if not found.
    """
    models = fetch_models_detailed(base_url)
    for m in models:
        if m.get("id") == model_id:
            return m
    return None
