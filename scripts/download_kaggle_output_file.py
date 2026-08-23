#!/usr/bin/env python3
"""Resume one private Kaggle kernel output file without printing its signed URL."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_kaggle_notebook as kaggle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="owner/kernel-slug")
    parser.add_argument("file", help="exact relative output file path")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--attempts", type=int, default=30)
    return parser.parse_args()


def signed_url(api: KaggleApi, kernel: str, filename: str) -> str:
    owner, slug, _ = api.parse_kernel_string(kernel)
    token = None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                if item.file_name == filename:
                    if not item.url:
                        raise RuntimeError(f"Kaggle returned no URL for {filename}")
                    return item.url
            token = response.next_page_token
            if not token:
                break
    raise FileNotFoundError(f"Kernel output does not contain {filename}")


def main() -> int:
    args = parse_args()
    if args.attempts < 1:
        raise ValueError("--attempts must be positive")
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    api = KaggleApi()
    api.authenticate()
    destination = args.destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.is_file() and destination.stat().st_size:
        destination.replace(partial)
    expected_size: int | None = None
    for attempt in range(1, args.attempts + 1):
        before = partial.stat().st_size if partial.exists() else 0
        print(f"download attempt {attempt}/{args.attempts}; resumed bytes={before:,}", flush=True)
        try:
            url = signed_url(api, args.kernel, args.file)
        except Exception as error:
            print(f"metadata request interrupted: {type(error).__name__}", flush=True)
            time.sleep(3)
            continue
        headers = {"Accept-Encoding": "identity"}
        if before:
            headers["Range"] = f"bytes={before}-"
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 90),
            ) as response:
                if before and response.status_code != 206:
                    raise RuntimeError(
                        f"Kaggle output server did not honor Range: HTTP {response.status_code}"
                    )
                if not before and response.status_code not in {200, 206}:
                    raise RuntimeError(
                        f"Kaggle output server returned HTTP {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if match:
                    response_start, _, response_total = map(int, match.groups())
                    if response_start != before:
                        raise RuntimeError(
                            f"Kaggle resumed at {response_start:,}, expected {before:,}"
                        )
                    expected_size = response_total
                elif not before:
                    content_length = response.headers.get("Content-Length")
                    expected_size = int(content_length) if content_length else None
                mode = "ab" if before else "wb"
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
        except Exception as error:
            print(f"stream interrupted: {type(error).__name__}", flush=True)
        after = partial.stat().st_size if partial.exists() else 0
        total_text = f"/{expected_size:,}" if expected_size is not None else ""
        print(f"downloaded bytes={after:,}{total_text}", flush=True)
        if expected_size is not None and after == expected_size:
            partial.replace(destination)
            print(f"complete: {destination} ({after:,} bytes)")
            return 0
        if after <= before:
            print("no progress", flush=True)
    raise RuntimeError(f"download did not complete after {args.attempts} attempts")


if __name__ == "__main__":
    raise SystemExit(main())
