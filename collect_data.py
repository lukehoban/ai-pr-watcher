#!/usr/bin/env python3
# PR‑tracker: counts Copilot / Codex PRs and saves data to CSV.
# Tracks merged PRs (not just approved ones)
# deps: requests

import csv
import datetime as dt
import re
import os
from pathlib import Path
import requests

# Basic headers for GitHub public API
HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "PR-Watcher", "Authorization": f"Bearer {os.environ['GITHUB_ACCESS_TOKEN']}"}

# Search queries - tracking merged PRs
Q = {
    "is:pr+head:copilot/": "copilot_total",
    "is:pr+head:copilot/+-is:draft": "copilot_ready",
    "is:pr+head:copilot/+is:merged": "copilot_merged",
    "is:pr+head:codex/": "codex_total",
    "is:pr+head:codex/+-is:draft": "codex_ready",
    "is:pr+head:codex/+is:merged": "codex_merged",
    "is:pr+head:cursor/": "cursor_total",
    "is:pr+head:cursor/+-is:draft": "cursor_ready",
    "is:pr+head:cursor/+is:merged": "cursor_merged",
    "is:pull-request+author:devin-ai-integration[bot]": "devin_total",
    "is:pull-request+author:devin-ai-integration[bot]+-is:draft": "devin_ready",
    "is:pull-request+author:devin-ai-integration[bot]+is:merged": "devin_merged",
    "is:pull-request+author:codegen-sh[bot]": "codegen_total",
    "is:pull-request+author:codegen-sh[bot]+-is:draft": "codegen_ready",
    "is:pull-request+author:codegen-sh[bot]+is:merged": "codegen_merged",
}


def collect_data():
    # Get data from GitHub API
    cnt = {}
    for query, key in Q.items():
        r = requests.get(
            f"https://api.github.com/search/issues?q={query}",
            headers=HEADERS,
            timeout=30,
        )
        print(r.status_code)
        print(r.text)
        r.raise_for_status()
        cnt[key] = r.json()["total_count"]

    # Save data to CSV
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y‑%m‑%d %H:%M:%S")
    row = [
        timestamp,
        cnt["copilot_total"],
        cnt["copilot_ready"],
        cnt["copilot_merged"],
        cnt["codex_total"],
        cnt["codex_ready"],
        cnt["codex_merged"],
        cnt["cursor_total"],
        cnt["cursor_ready"],
        cnt["cursor_merged"],
        cnt["devin_total"],
        cnt["devin_ready"],
        cnt["devin_merged"],
        cnt["codegen_total"],
        cnt["codegen_ready"],
        cnt["codegen_merged"],
    ]

    csv_file = Path("data.csv")
    is_new_file = not csv_file.exists()
    with csv_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(
                [
                    "timestamp",
                    "copilot_total",
                    "copilot_ready",
                    "copilot_merged",
                    "codex_total",
                    "codex_ready",
                    "codex_merged",
                    "cursor_total",
                    "cursor_ready",
                    "cursor_merged",
                    "devin_total",
                    "devin_ready",
                    "devin_merged",
                    "codegen_total",
                    "codegen_ready",
                    "codegen_merged",
                ]
            )
        writer.writerow(row)

    return csv_file


def update_html_with_latest_data():
    """Update the HTML file with the latest statistics from the chart data."""
    # The HTML will be updated by JavaScript automatically when chart-data.json loads
    # This is a placeholder for any additional HTML updates needed
    html_file = Path("docs/index.html")
    if not html_file.exists():
        print("HTML file not found, skipping HTML update")
        return

    # Update the last updated timestamp in the HTML
    html_content = html_file.read_text()

    # Get current timestamp in the format used in the HTML
    now = dt.datetime.now(dt.UTC)
    timestamp_str = now.strftime("%B %d, %Y %H:%M UTC")

    # Update the timestamp in the HTML
    updated_html = re.sub(
        r'<span id="last-updated">[^<]*</span>',
        f'<span id="last-updated">{timestamp_str}</span>',
        html_content,
    )

    html_file.write_text(updated_html)
    print(f"Updated HTML timestamp to: {timestamp_str}")


if __name__ == "__main__":
    collect_data()
    update_html_with_latest_data()
    print("Data collection complete. To generate chart, run generate_chart.py")
