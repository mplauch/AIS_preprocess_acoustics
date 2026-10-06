import time
import re
import random
from urllib.parse import urljoin, urlparse

from curl_cffi import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------- CONFIG ----------------
YEAR = 2023
TYPE = "Ferry"
INPUT_CSV = f"/Users/maximilianlauch/Documents/AIS/AIS_Diagnostic_Checks/mmsi-length_database/{YEAR}_comparison_output/{YEAR}_uncorrected_mmsi_lists/mmsi_list_{TYPE}.csv"
OUT_CSV   = f"/Users/maximilianlauch/Documents/AIS/AIS_Diagnostic_Checks/2023_Scrub/myshiptracking_lookup_{TYPE}.csv"

BASE = "https://www.myshiptracking.com"
SLEEP_SEC = 3.0
TIMEOUT = 25
DEBUG_SAVE_HTML = True   # saves the first vessel page so you can inspect labels
# ---------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Labels as they appear on myshiptracking vessel pages.
# If a column comes back empty, check the saved debug HTML and edit these.
TARGET_LABELS = {
    "Ship Type": ["vessel_type"],
    "Size": ["Size", "Length / Beam", "Dimensions"],
}


def extract_type_from_blurb(soup, mmsi):
    """'(IMO: 9318591, MMSI: 538009026) is a Bulk Carrier that was built...' -> 'Bulk Carrier'"""
    text = normalize_text(soup.get_text(" ", strip=True))
    patterns = [
        rf"MMSI:\s*{mmsi}\s*\)\s*is an?\s+(.+?)(?=\s+that was built|\.\s|\s+It's sailing)",
        r"\)\s*is an?\s+(.+?)\s+that was built",   # looser fallback
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def labels_match(a: str, b: str) -> bool:
    # Case-insensitive, ignores whitespace and trailing colons ("Type:" == "Type")
    clean = lambda x: normalize_text(x).rstrip(":").strip().lower()
    return clean(a) == clean(b)


def extract_value_from_table_rows(soup, label):
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        if labels_match(cells[0].get_text(" ", strip=True), label):
            v = normalize_text(cells[1].get_text(" ", strip=True))
            return v or None
    return None


def extract_value_by_nearby_text(soup, label):
    candidates = soup.find_all(string=lambda t: t and labels_match(t, label))
    for cand in candidates:
        el = cand.parent
        sib = el.find_next_sibling()
        if sib:
            txt = normalize_text(sib.get_text(" ", strip=True))
            if txt:
                return txt
        parent = el.parent
        if parent:
            sib2 = parent.find_next_sibling()
            if sib2:
                txt = normalize_text(sib2.get_text(" ", strip=True))
                if txt:
                    return txt
    return None


def extract_field(soup, label_options):
    for label in label_options:
        val = extract_value_from_table_rows(soup, label)
        if val is None:
            val = extract_value_by_nearby_text(soup, label)
        if val:
            return val
    return None


def name_from_url(url):
    """'.../vessels/saga-monal-mmsi-477878000-imo-0' -> 'SAGA MONAL'"""
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    m = re.match(r"(.+?)-mmsi-\d+", slug)
    if m and m.group(1) != "vessel":
        return m.group(1).replace("-", " ").upper()
    return None


def extract_vessel_name(soup, url):
    h1 = soup.find("h1")
    if h1:
        name = normalize_text(h1.get_text(" ", strip=True))
        if name:
            return name
    return name_from_url(url)


def parse_length_m(size_str):
    """'183 x 32 m' or '183 m x 32 m' -> 183.0"""
    if not size_str:
        return None
    s = size_str.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m)?\s*[x×/]", s)
    return float(m.group(1)) if m else None


def find_vessel_url(session, mmsi):
    """Search the site by MMSI and return the vessel page URL (with name slug)."""
    try:
        r = session.get(f"{BASE}/vessels", params={"name": mmsi}, timeout=TIMEOUT)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                if re.search(rf"mmsi-{mmsi}(?:-|$)", a["href"]):
                    return urljoin(BASE, a["href"])
    except requests.RequestException:
        pass
    # Fallback: placeholder slug; many sites of this type key off the MMSI only
    return f"{BASE}/vessels/vessel-mmsi-{mmsi}-imo-0"


def query_myshiptracking_by_mmsi(session, mmsi, save_html=False):
    url = find_vessel_url(session, mmsi)
    result = {
        "MMSI": mmsi,
        "Vessel Name": None,
        "Ship Type": None,
        "Size": None,
        "Length (m)": None,
        "status": None,
        "url": url,
    }

    time.sleep(1 + random.uniform(0, 1))  # small gap between search and detail request
    r = session.get(url, timeout=TIMEOUT)
    result["url"] = r.url  # keep the final URL if the site redirected

    if r.status_code != 200:
        result["status"] = f"http_{r.status_code}"
        return result

    if save_html:
        with open(f"debug_myshiptracking_{mmsi}.html", "w", encoding="utf-8") as f:
            f.write(r.text)

    soup = BeautifulSoup(r.text, "html.parser")
    result["Vessel Name"] = extract_vessel_name(soup, r.url)
    result["Ship Type"] = extract_type_from_blurb(soup, mmsi)
    result["Size"] = extract_field(soup, TARGET_LABELS["Size"])
    result["Length (m)"] = parse_length_m(result["Size"])

    result["status"] = "ok"
    return result


# --------------- RUN ----------------
if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, header=0)
    mmsis = df['mmsi'].tolist()

    rows = []
    with requests.Session(impersonate="chrome") as session:
        for i, mmsi in enumerate(tqdm(mmsis)):
            try:
                rows.append(query_myshiptracking_by_mmsi(
                    session, mmsi, save_html=(DEBUG_SAVE_HTML and i == 0)))
            except Exception as e:
                rows.append({"MMSI": mmsi, "status": f"error: {e}"})
            time.sleep(SLEEP_SEC + random.uniform(0.5, 1.5))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")
