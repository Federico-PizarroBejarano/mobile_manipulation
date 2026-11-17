#!/usr/bin/env python3
"""lerobot-train wrapper that keeps observation.* width-1 vectors as vectors.

Needed for ``env_state=zeros`` (shape (1,)) datasets; see
``mm_utils.lerobot_hf_features``.
"""

from __future__ import annotations

import sys

from mm_utils.lerobot_hf_features import patch_lerobot_len1_obs_features


def main(argv=None) -> int:
    patch_lerobot_len1_obs_features()
    # Match ``lerobot-train`` console script argv handling.
    if argv is not None:
        sys.argv = [sys.argv[0], *list(argv)]
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
