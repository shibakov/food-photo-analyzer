"""Utility functions."""

import json
import re


def extract_json(text: str) -> dict:
    """
    Clean Markdown, ```json, comments.
    Return Json object.
    """
    if not text:
        raise ValueError("Empty model output")

    # Remove ```json ... ```
    text = re.sub(r"```.*?```", "", text, flags=re.S)

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object detected")

    cleaned = text[start:end+1]

    return json.loads(cleaned)
