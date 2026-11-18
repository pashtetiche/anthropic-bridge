import argparse
import os

import uvicorn

from .logging import setup_access_logging, setup_debug_log_file
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic Bridge Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    copilot_token = os.environ.get("GITHUB_COPILOT_TOKEN", "")

    debug_log_path = setup_debug_log_file()
    log_config = setup_access_logging()

    app = create_app(
        openrouter_api_key=api_key or None,
        copilot_token=copilot_token or None,
    )

    print(f"Starting Anthropic Bridge on {args.host}:{args.port}")
    if debug_log_path:
        print(f"  Debug log: {debug_log_path} (latest: {debug_log_path.parent / 'anthropic-bridge-debug-latest.log'})")
    print("  OpenAI: openai/* models")
    if copilot_token:
        print("  Copilot: copilot/* models")
    else:
        print("  Copilot: disabled (set GITHUB_COPILOT_TOKEN)")
    if api_key:
        print("  OpenRouter: openrouter/* models")
    else:
        print("  OpenRouter: disabled (set OPENROUTER_API_KEY)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", log_config=log_config)


if __name__ == "__main__":
    main()
