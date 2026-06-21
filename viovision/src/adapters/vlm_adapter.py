"""
VLM adjudication adapter. Guide section 2.F is explicit: this is
prompt-only, never trained. It's the explainability/review layer that
low-confidence classifier outputs route into (seatbelt, OCR, signal-state
at night all feed here per guide section 7).

Uses the Anthropic API directly since that's already configured in this
environment. Swap the model/provider here if your hackathon stack uses
something else — the adapter shape (adjudicate()) is what matters, not
which VLM backs it.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import requests

from src.adapters.schema import ViolationType, VLMVerdict

ADJUDICATION_PROMPT_TEMPLATE = """\
You are reviewing a single cropped frame from a traffic-violation \
detection pipeline. The automated system flagged this as a candidate \
"{violation_type}" violation, but confidence was below the auto-file \
threshold, so it has been routed to you for review.

Additional context from the pipeline: {context}

Look at the image and answer:
1. Does the image support this violation? (yes/no)
2. One sentence justification.

Respond ONLY as JSON: {{"violation_confirmed": true/false, "justification": "..."}}
"""


def _encode_crop_to_base64(crop_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", crop_bgr)
    if not success:
        raise ValueError("Failed to encode crop to JPEG for VLM adjudication.")
    return base64.b64encode(buffer).decode("utf-8")


class ClaudeVLMAdjudicatorAdapter:
    """Satisfies VLMAdjudicatorAdapter using the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.anthropic.com/v1/messages"

    def adjudicate(self, crop: np.ndarray, candidate_violation: ViolationType,
                    context: str) -> VLMVerdict:
        image_b64 = _encode_crop_to_base64(crop)
        prompt = ADJUDICATION_PROMPT_TEMPLATE.format(
            violation_type=candidate_violation, context=context
        )

        response = requests.post(
            self.endpoint,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 300,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg",
                            "data": image_b64,
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        text_block = next(
            (b["text"] for b in payload["content"] if b["type"] == "text"), "{}"
        )

        import json
        try:
            parsed = json.loads(text_block)
        except json.JSONDecodeError:
            # Model didn't follow the JSON-only instruction; fail safe
            # rather than crash the pipeline on a malformed response.
            parsed = {"violation_confirmed": False,
                       "justification": "VLM response unparseable; treat as unconfirmed."}

        return VLMVerdict(
            violation_confirmed=bool(parsed.get("violation_confirmed", False)),
            justification=str(parsed.get("justification", "")),
            violation_type=candidate_violation,
        )
