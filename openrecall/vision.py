"""
vision.py — AI vision analysis of screenshots via OpenRouter.

Replaces the old OCR-only approach with a vision LLM that understands
not just text on screen, but context, activity, and intent.
"""

import base64
import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)

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


def analyze_screenshot(image: Image.Image) -> str:
    """
    Sends a screenshot to an OpenRouter vision LLM and returns a rich
    natural-language description of what is happening on screen.

    Returns an empty string if the API key is not configured or the call fails.
    """
    from openrecall.config import openrouter_api_key, vision_model

    if not openrouter_api_key:
        logger.warning(
            "OPENROUTER_API_KEY is not set. "
            "Pass --openrouter-api-key or set the env var to enable vision analysis."
        )
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

        # Resize image to reduce payload size and API cost
        img = image.copy()
        img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)

        # Encode as JPEG (much smaller than lossless WebP for API calls)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model=vision_model,
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
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        description = response.choices[0].message.content.strip()
        logger.info("Vision analysis complete: %s...", description[:80])
        return description

    except Exception as exc:
        logger.error("Vision analysis failed: %s", exc)
        return ""
