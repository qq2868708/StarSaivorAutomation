"""Extract API URLs from GameKee JS bundles."""
import re
import urllib.request
import sys

URLS = [
    "https://cdnstatic.gamekee.com/wiki/spa/apps/web/client/dist/js/runtime~main.c55246c2.js",
    "https://cdnstatic.gamekee.com/wiki/spa/apps/web/client/dist/js/main.d0eb8d57.js",
    "https://cdnstatic.gamekee.com/wiki/spa/apps/web/client/dist/js/8333.354b436d.js",
]

for url in URLS:
    print(f"\n=== {url.split('/')[-1]} ===")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error: {e}")
        continue

    # Find all gamekee.com URLs
    matches = re.findall(r'https?://[a-zA-Z0-9.-]+\.gamekee\.com[^\s"\')\],;]+', data)
    for m in sorted(set(matches)):
        print(m)

    # Find baseURL patterns
    base_matches = re.findall(r'baseURL["\']?\s*[=:]\s*["\'][^"\']+["\']', data)
    for m in sorted(set(base_matches)):
        print(f"  baseURL: {m}")

    # Find api-cdn references
    cdn = re.findall(r'[a-z]+-cdn\.[a-z]+\.com[^\s"\')\],;]+', data)
    for m in sorted(set(cdn)):
        print(f"  cdn: {m}")
