"""
screenshot.py — Screen capture loop.

Captures screenshots at regular intervals, skips frames that haven't
changed, then sends changed frames to the vision LLM for analysis
before storing in ChromaDB.
"""

import os
import time
from typing import List

import mss
import numpy as np
from PIL import Image

from openrecall.config import screenshots_path, args, capture_interval
from openrecall.database import insert_entry
from openrecall.vision import analyze_screenshot
from openrecall.utils import (
    get_active_app_name,
    get_active_window_title,
    is_user_active,
)


def mean_structured_similarity_index(
    img1: np.ndarray, img2: np.ndarray, L: int = 255
) -> float:
    """Mean Structural Similarity Index between two RGB images."""
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    def rgb2gray(img: np.ndarray) -> np.ndarray:
        return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

    g1, g2 = rgb2gray(img1), rgb2gray(img2)
    mu1, mu2 = np.mean(g1), np.mean(g2)
    sigma1_sq, sigma2_sq = np.var(g1), np.var(g2)
    sigma12 = np.mean((g1 - mu1) * (g2 - mu2))
    return float(
        ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2))
        / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    )


def is_similar(
    img1: np.ndarray, img2: np.ndarray, threshold: float = 0.9
) -> bool:
    """True if two screenshots are visually similar (skip processing)."""
    return mean_structured_similarity_index(img1, img2) >= threshold


def take_screenshots() -> List[np.ndarray]:
    """Capture all connected monitors (or primary only) as RGB numpy arrays."""
    screenshots: List[np.ndarray] = []
    with mss.mss() as sct:
        monitor_indices = (
            [1] if args.primary_monitor_only else range(1, len(sct.monitors))
        )
        for i in monitor_indices:
            if i < len(sct.monitors):
                sct_img = sct.grab(sct.monitors[i])
                # BGRA → RGB
                screenshots.append(np.array(sct_img)[:, :, [2, 1, 0]])
    return screenshots


def record_screenshots_thread() -> None:
    """
    Main capture loop — runs in a background thread.

    For each changed screenshot:
      1. Save as WebP (lossless, for the timeline viewer)
      2. Send to vision LLM for semantic description
      3. Store description + metadata in ChromaDB
    """
    # Prevent a noisy warning from the HuggingFace tokenizers library
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    last_screenshots = take_screenshots()

    while True:
        if not is_user_active():
            time.sleep(capture_interval)
            continue

        current_screenshots = take_screenshots()

        # Handle monitor count change gracefully
        if len(last_screenshots) != len(current_screenshots):
            last_screenshots = current_screenshots
            time.sleep(capture_interval)
            continue

        for i, current in enumerate(current_screenshots):
            last = last_screenshots[i]

            if is_similar(current, last):
                continue  # Screen hasn't changed — skip

            # Update reference frame
            last_screenshots[i] = current

            # Save screenshot for the timeline viewer
            image = Image.fromarray(current)
            timestamp = int(time.time())
            filename = f"{timestamp}_{i}.webp"
            filepath = os.path.join(screenshots_path, filename)
            image.save(filepath, format="webp", lossless=True)

            # Vision LLM: understand what's on screen
            description = analyze_screenshot(image)

            if description.strip():
                insert_entry(
                    text=description,
                    timestamp=timestamp,
                    app=get_active_app_name() or "Unknown App",
                    title=get_active_window_title() or "Unknown Title",
                    filename=filename,
                )

        time.sleep(capture_interval)
