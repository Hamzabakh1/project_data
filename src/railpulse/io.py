"""Download the SNCF Intercités official CSV export."""
from pathlib import Path
from urllib.request import Request, urlopen

SNCF_URL = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-intercites/exports/csv"

def download_sncf(path="data/regularite-mensuelle-intercites.csv"):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(SNCF_URL, headers={"User-Agent": "railpulse-oncf/0.1"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    if len(content) < 100:
        raise ValueError("SNCF export unexpectedly empty")
    target.write_bytes(content)
    return target
