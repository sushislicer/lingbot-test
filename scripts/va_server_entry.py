#!/usr/bin/env python3
"""Thin wrapper to start LingBot-VA websocket server with runtime config overrides.

We treat upstream `lingbot-va` as read-only. Upstream config
[`lingbot-va/wan_va/configs/va_robotwin_cfg.py`](lingbot-va/wan_va/configs/va_robotwin_cfg.py:1)
contains a placeholder model path.

This entrypoint patches `VA_CONFIGS[config_name].wan22_pretrained_model_name_or_path`
from the environment variable `WAN22_MODEL_PATH` before calling
[`wan_va.wan_va_server.run()`](lingbot-va/wan_va/wan_va_server.py:667).

This file is launched under `torch.distributed.run` so it executes on every rank.
"""

import argparse
import os
import sys
from pathlib import Path


def _maybe_add_lingbot_va_to_syspath() -> None:
    root = os.environ.get("LINGBOT_VA_ROOT")
    if not root:
        return
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    _maybe_add_lingbot_va_to_syspath()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="robotwin")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    args = parser.parse_args()

    wan22_model_path = os.environ.get("WAN22_MODEL_PATH")
    if not wan22_model_path:
        raise SystemExit(
            "WAN22_MODEL_PATH is not set. Example: "
            "WAN22_MODEL_PATH=/workspace/lingbot-test/checkpoints/lingbot-va-posttrain-robotwin"
        )
    if not Path(wan22_model_path).exists():
        raise SystemExit(f"WAN22_MODEL_PATH does not exist: {wan22_model_path}")

    # Import after path setup.
    from wan_va.configs import VA_CONFIGS  # type: ignore

    if args.config_name not in VA_CONFIGS:
        raise SystemExit(
            f"Unknown config-name {args.config_name!r}. Available: {sorted(VA_CONFIGS.keys())}"
        )

    VA_CONFIGS[args.config_name].wan22_pretrained_model_name_or_path = wan22_model_path

    from wan_va.wan_va_server import run  # type: ignore

    run(args)


if __name__ == "__main__":
    main()

