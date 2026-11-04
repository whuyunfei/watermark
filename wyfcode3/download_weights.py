#!/usr/bin/env python3
"""Download Stable Diffusion 2.1 weights into ringid_v4_project/models.

First run of generate.py / detect.py / eval_quick.py calls this automatically.
"""

from ringid.pipeline import CACHE_DIR, MODEL_ID, download_weights


def main() -> None:
    print(f"Model: {MODEL_ID}")
    print(f"Cache: {CACHE_DIR}")
    path = download_weights()
    print(f"Ready: {path}")


if __name__ == "__main__":
    main()
