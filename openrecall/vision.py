"""
vision.py — AI vision analysis of screenshots via OpenRouter.

Sends each screenshot through a configurable fallback chain of vision
models. On a rate-limit hit (429) we skip immediately to the next model
in the chain; on transient errors we retry the same model once before
moving on. The capture loop is never blocked for more than ~total chain
length seconds.
"""

import base64
import io
import logging
import time

from PIL import Image

logger = logging.getLogger(__name__)

# Per-model retry policy — only retry on transient errors (504, network,
# empty content), not on 429s. The chain itself is the real fallback.
MAX_RETRIES_PER_MODEL = 1     # initial + 1 retry on transient failures
RETRY_BACKOFF_SECONDS = 5

VISION_PROMPT = (
    "Analyze this screenshot and describe:\n"
    "1. What application(s) are open and what the user is doing\n"
    "2. Key visible text (titles, messages, code, documents, URLs)\n"
    "3. The overall activity context (e.g. 'writing an email to John', "
    "'browsing Reddit', 'coding Python in VS Code', 'watching a YouTube video')\n\n"
    "Be concise but thorough. Focus on what would help someone recall "
    "what they were doing at this exact moment."
)

# Max resolution sent to the API — keeps costs low without losing meaningful detail
MAX_WIDTH = 1280
MAX_HEIGHT = 720


def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristic: did this exception come from a 429 response?"""
    s = str(exc)
    return "429" in s or "rate" in s.lower()


def _call_model(client, model: str, image_b64: str) -> str:
    """
    One vision call against a single model. Returns the description, or
    raises if the model returned no usable content.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
        max_tokens=500,
    )

    msg = response.choices[0].message if response.choices else None
    if msg is None:
        raise ValueError("No choices in vision API response")

    # Reasoning-style models put text in `reasoning` instead of `content`
    description = (
        getattr(msg, "content", None)
        or getattr(msg, "reasoning", None)
        or ""
    ).strip()

    if not description:
        raise ValueError("Vision model returned empty content")

    return description


def analyze_screenshot(image: Image.Image) -> str:
    """
    Sends a screenshot through the configured fallback chain of vision
    models. Returns the first successful description, or empty string if
    every model in the chain fails.
    """
    from openrecall.config import openrouter_api_key, vision_models

    if not openrouter_api_key:
        logger.warning(
            "OPENROUTER_API_KEY is not set. "
            "Pass --openrouter-api-key or set the env var to enable vision analysis."
        )
        return ""

    if not vision_models:
        logger.warning("No vision models configured.")
        return ""

    try:
        import openai

        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/sammilvid/openrecall",
                "X-Title": "OpenRecall",
            },
        )

        # Resize and JPEG-encode once — reuse for every model in the chain
        img = image.copy()
        img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        last_exc: Exception | None = None

        for model_idx, model in enumerate(vision_models):
            logger.debug("Trying vision model %d/%d: %s",
                         model_idx + 1, len(vision_models), model)

            for attempt in range(MAX_RETRIES_PER_MODEL + 1):
                try:
                    description = _call_model(client, model, image_b64)
                    logger.info("Vision analysis OK [%s]: %s...",
                                model, description[:80])
                    return description

                except Exception as exc:
                    last_exc = exc

                    if _is_rate_limit_error(exc):
                        # Don't waste time retrying a rate-limited model —
                        # jump straight to the next one in the chain
                        logger.info("Model %s rate-limited, falling back", model)
                        break

                    if attempt < MAX_RETRIES_PER_MODEL:
                        logger.debug(
                            "Model %s attempt %d failed (%s) — retrying in %ds",
                            model, attempt + 1, exc, RETRY_BACKOFF_SECONDS,
                        )
                        time.sleep(RETRY_BACKOFF_SECONDS)
                        continue

                    logger.warning("Model %s exhausted retries: %s", model, exc)
                    break

        logger.error(
            "All %d vision models failed. Last error: %s",
            len(vision_models), last_exc,
        )
        return ""

    except Exception as exc:
        logger.error("Vision setup failed: %s", exc)
        return ""
