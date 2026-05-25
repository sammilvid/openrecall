import os
import sys
import argparse

parser = argparse.ArgumentParser(description="OpenRecall")

parser.add_argument(
    "--storage-path",
    default=None,
    help="Path to store the screenshots and database",
)

parser.add_argument(
    "--primary-monitor-only",
    action="store_true",
    help="Only record the primary monitor",
    default=False,
)

parser.add_argument(
    "--openrouter-api-key",
    default=os.environ.get("OPENROUTER_API_KEY", ""),
    help="OpenRouter API key for vision analysis (or set OPENROUTER_API_KEY env var)",
)

parser.add_argument(
    "--vision-model",
    default=(
        "google/gemma-4-26b-a4b-it:free,"
        "nvidia/nemotron-nano-12b-v2-vl:free,"
        "google/gemma-4-31b-it:free,"
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    ),
    help=(
        "OpenRouter vision model(s) to use for screenshot analysis. "
        "Comma-separated list — tried in order, falling back on rate limits or errors. "
        "Defaults to a chain of free vision models."
    ),
)

parser.add_argument(
    "--capture-interval",
    type=int,
    default=60,
    help="Seconds between screen capture checks (default: 60). Higher = fewer API calls, less rate-limit pressure.",
)

args = parser.parse_args()

openrouter_api_key: str = args.openrouter_api_key
vision_models: list[str] = [m.strip() for m in args.vision_model.split(",") if m.strip()]
capture_interval: int = args.capture_interval


def get_appdata_folder(app_name="openrecall"):
    if sys.platform == "win32":
        appdata = os.getenv("APPDATA")
        if not appdata:
            raise EnvironmentError("APPDATA environment variable is not set.")
        path = os.path.join(appdata, app_name)
    elif sys.platform == "darwin":
        home = os.path.expanduser("~")
        path = os.path.join(home, "Library", "Application Support", app_name)
    else:
        home = os.path.expanduser("~")
        path = os.path.join(home, ".local", "share", app_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if args.storage_path:
    appdata_folder = args.storage_path
    screenshots_path = os.path.join(appdata_folder, "screenshots")
    db_path = os.path.join(appdata_folder, "recall.db")
else:
    appdata_folder = get_appdata_folder()
    db_path = os.path.join(appdata_folder, "recall.db")
    screenshots_path = os.path.join(appdata_folder, "screenshots")

if not os.path.exists(screenshots_path):
    try:
        os.makedirs(screenshots_path)
    except:
        pass
