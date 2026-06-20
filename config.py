"""Config loading + global seeding for DUALMIND.

Single source of truth for reading config.yaml and seeding every RNG so the whole
pipeline is reproducible (the seed is logged in the run manifest).
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import yaml

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def _bootstrap_env() -> None:
    """Load .env and make TLS work behind a corporate/WARP MITM proxy.

    On machines that intercept TLS (e.g. Cloudflare WARP), Python's bundled
    certifi store doesn't trust the injected root, so the anthropic SDK's httpx
    client fails verification. If a system-trust PEM bundle is configured via
    SSL_CERT_FILE (set it in .env), mirror it into the vars httpx/requests read.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    bundle = os.environ.get("SSL_CERT_FILE")
    if bundle and os.path.exists(bundle):
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)


def load_config(path: str = _DEFAULT_PATH) -> Dict[str, Any]:
    _bootstrap_env()
    with open(path, "r", encoding="utf-8") as f:
        # safe_load returns None for an empty/comment-only file.
        return yaml.safe_load(f) or {}


def seed_everything(seed: int) -> None:
    """Seed stdlib + numpy (if present) + torch (if present)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_and_seed(path: str = _DEFAULT_PATH) -> Dict[str, Any]:
    cfg = load_config(path)
    seed_everything(int(cfg.get("seed", 42)))
    return cfg
