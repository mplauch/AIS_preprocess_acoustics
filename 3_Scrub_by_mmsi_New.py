"""
Scrub the "to_scrub" ships from the flagged csv on myshiptracking.com.

INPUT   2024_flagged.csv (output of Compare_mmsi_databases_V3.py)

OUTPUT  2024_scrub_results.csv   raw scrape results (saved as it goes, used to resume)
        2024_flagged_merged.csv  2024_flagged.csv with the scraped data standardized and merged in:
                                 - Type mapped to the standard vocabulary, Name uppercased, Length/Width in metres
                                 - applied field by field, only where the scraped value is usable
                                 - flag becomes "scraped" for ships that received scraped data;
                                   ships that could not be scraped stay "to_scrub"
                                 The input 2024_flagged.csv is never modified.
"""
import time
import re
import random
from pathlib import Path
from urllib.parse import urljoin, urlparse

from curl_cffi import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

DIR = Path("C:/Users/ZHANGD/Desktop/AIS_preprocess_acoustics/output")
INPUT_CSV = DIR / "2015_flagged.csv"
RESULTS_CSV = DIR / "2015_scrub_results.csv"   # raw scrape results, saved as it goes (used to resume)
OUTPUT_CSV = DIR / "2015_flagged_merged.csv"   # flagged list with the standardized scrape data merged in

BASE = "https://www.myshiptracking.com"
SLEEP_SEC = 3.0
TIMEOUT = 25
SAVE_EVERY = 25
DEBUG_SAVE_HTML = False   

MMSI_COL = "mmsi"
FLAG_COL = "flag"



# Labels as they appear on myshiptracking vessel pages.
# If a column comes back empty, check the saved debug HTML and edit these.
TARGET_LABELS = {"Size": ["Size", "Length / Beam", "Dimensions"],}

# Scraped dimensions outside these ranges (metres) are treated as bad and ignored
LENGTH_RANGE = (5, 500)
WIDTH_RANGE = (1, 100)

# myshiptracking.com "Ship Type" labels -> the standard type names used in the AIS files.
SCRAPE_TYPE_MAP = {
    # Cargo
    'Cargo': 'Cargo', 'Bulk Carrier': 'Cargo', 'General Cargo': 'Cargo', 'Bulker': 'Cargo',
    'Cargo A': 'Cargo', 'Cargo B': 'Cargo', 'Cargo C': 'Cargo', 'Cargo D': 'Cargo',
    'Reefer': 'Cargo', 'Livestock Carrier': 'Cargo', 'Ore Carrier': 'Cargo',
    'Heavy Lift Vessel': 'Cargo', 'Heavy Load Carrier': 'Cargo', 'Cargo/Container Ship': 'Cargo',
    'Container Ship': 'Containership',
    'Vehicles Carrier': 'VehicleCarrier', 'Rail/Vehicles Carrier': 'VehicleCarrier',
    # Tanker
    'Tanker': 'Tanker', 'Oil/Chemical Tanker': 'Tanker', 'Crude Oil Tanker': 'Tanker',
    'Oil Products Tanker': 'Tanker', 'Chemical Tanker': 'Tanker', 'Lpg Tanker': 'Tanker',
    'Asphalt/Bitumen Tanker': 'Tanker', 'Tanker B': 'Tanker', 'Tanker C': 'Tanker', 'Tanker D': 'Tanker',
    # Recreational
    'Pleasure Craft': 'Recreational', 'Yacht': 'Recreational', 'Sailing': 'Recreational',
    # Fishing
    'Fishing': 'Fishing', 'Fishing Vessel': 'Fishing', 'Trawler': 'Fishing', 'Fish Factory': 'Fishing',
    # Tug
    'Tug': 'Tug', 'Towing': 'Tug', 'Articulated Pusher Tug': 'Tug', 'Towing > 200m OR Breadth > 25m': 'Tug',
    # Passenger / Ferry
    'Passenger': 'Passenger', 'Passengers Ship': 'Passenger', 'Ro-Ro/Passenger Ship': 'Ferry',
    # Naval / Government
    'Military ops': 'Naval', 'Noncombatant ship': 'Naval', 'Replenishment Vessel': 'Naval',
    'Search and Rescue vessel': 'Government', 'Law Enforcement': 'Government',
    'Anti-pollution equipment': 'Government', 'Pollution Control Vessel': 'Government',
    'Research/Survey Vessel': 'Government', 'Fishery Research Vessel': 'Government',
    'Icebreaker': 'Government',
    # Other (work / offshore / misc)
    'Other Type': 'Other', 'Anchor Handling Vessel': 'Other', 'Offshore Supply Ship': 'Other',
    'Cable Layer': 'Other', 'Dredging or underwater ops': 'Other', 'Diving ops': 'Other',
    'Buoy-laying Vessel': 'Other', 'Pipelay Crane Vessel': 'Other', 'Multi Purpose Offshore Vessel': 'Other',
    'Trailing Suction Hopper Dredger': 'Other', 'Pilot Vessel': 'Other', 'Port Tender': 'Other',
    'Wing in ground': 'Other', 'Navigation Aid': 'Other',
}

# Scraped labels that just mean "no information" (ignored quietly)
SCRAPE_TYPE_EMPTY = {'Not available', '---', 'Reserved', 'Reserved for future use', 'Spare - Local Vessel',
                     'Base Station'}
# A scraped generic "Other" never overrides a specific Type the ship already has
WEAK_TYPES = {'Other'}

PLACEHOLDER_NAMES = {"", "UNKOWN NAME", "UNKNOWN NAME", "---", "NOT AVAILABLE"}

MERGE_FIELDS = ["Type", "Name", "Length", "Width"]


# ---------------- scraping functions ----------------
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
    # ignores whitespace and trailing punctuation, case insensitive ("Type:" == "Type")
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

    time.sleep(1 + random.uniform(0, 1))  # small gap between search and detail request, keeps from getting blocked
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


def load_results(path):
    """Previously scraped results as {mmsi: record}, or {} on a first run."""
    if not path.exists():
        return {}
    prev = pd.read_csv(path)
    return {int(r["MMSI"]): r for r in prev.to_dict("records")}


def save_results(results, path):
    pd.DataFrame(list(results.values())).to_csv(path, index=False)


def scrape(mmsis, results):
    """Scrape every mmsi not already done. Saves progress as it goes, even if interrupted."""
    todo = [m for m in mmsis if results.get(m, {}).get("status") not in ("ok", "http_404")]
    print(f"{len(mmsis)} ships to scrub, {len(mmsis) - len(todo)} already done, {len(todo)} to scrape now")

    with requests.Session(impersonate="chrome") as session:
        try:
            for i, mmsi in enumerate(tqdm(todo)):
                try:
                    results[mmsi] = query_myshiptracking_by_mmsi(
                        session, mmsi, save_html=(DEBUG_SAVE_HTML and i == 0))
                except Exception as e:
                    results[mmsi] = {"MMSI": mmsi, "status": f"error: {e}"}
                if (i + 1) % SAVE_EVERY == 0:
                    save_results(results, RESULTS_CSV)
                time.sleep(SLEEP_SEC + random.uniform(0.5, 1.5))
        finally:
            save_results(results, RESULTS_CSV)


# ---------------- standardize + merge ----------------
def parse_width_m(size_str):
    """'183 x 32 m' -> 32.0"""
    if not isinstance(size_str, str):
        return float("nan")
    m = re.search(r"[x×/]\s*(\d+(?:[.,]\d+)?)", size_str)
    return float(m.group(1).replace(",", ".")) if m else float("nan")


def standardize_results(results_csv):

    r = pd.read_csv(results_csv)
    stats = {"scraped rows": len(r)}
    r["mmsi"] = pd.to_numeric(r["MMSI"], errors="coerce").round().astype("Int64")
    r = r[r["mmsi"].notna() & (r["status"] == "ok")].drop_duplicates("mmsi", keep="last")
    stats["page loaded (status ok)"] = len(r)

    # the page that was fetched must belong to this MMSI
    same = [bool(re.search(rf"mmsi-{m}(?:-|$)", str(u))) for m, u in zip(r["mmsi"], r["url"])]
    r = r[same]
    stats["URL matches MMSI"] = len(r)

    std = pd.DataFrame(index=pd.Index(r["mmsi"].astype("int64").values, name="mmsi"))

    labels = pd.Series(r["Ship Type"].astype("string").str.strip().values, index=std.index)
    std["Type"] = labels.map(SCRAPE_TYPE_MAP)
    std["Type_weak"] = std["Type"].isin(WEAK_TYPES)
    unmapped = labels[std["Type"].isna() & labels.notna() & ~labels.isin(SCRAPE_TYPE_EMPTY)]
    stats["unmapped types"] = unmapped.value_counts().to_dict()

    name = pd.Series(r["Vessel Name"].astype("string").str.strip().str.upper().values, index=std.index)
    std["Name"] = name.where(~name.isin(PLACEHOLDER_NAMES))

    length = pd.Series(pd.to_numeric(r["Length (m)"], errors="coerce").values, index=std.index)
    width = pd.Series(r["Size"].map(parse_width_m).values, index=std.index)
    stats["length out of range (ignored)"] = int((length.notna() & ~length.between(*LENGTH_RANGE)).sum())
    stats["width out of range (ignored)"] = int((width.notna() & ~width.between(*WIDTH_RANGE)).sum())
    std["Length"] = length.where(length.between(*LENGTH_RANGE))
    std["Width"] = width.where(width.between(*WIDTH_RANGE))
    return std, stats


def merge_into_flagged(flagged, std):

    out = flagged.copy()
    out[MMSI_COL] = pd.to_numeric(out[MMSI_COL], errors="coerce").round().astype("Int64")
    for c in ("Type", "Name"):
        out[c] = out[c].astype("object")

    todo = (out[FLAG_COL] == "to_scrub") & out[MMSI_COL].isin(std.index)
    idx = out.index[todo]
    new = std.reindex(out.loc[idx, MMSI_COL].astype("int64").values)
    new.index = idx

    applied = pd.Series(False, index=out.index)
    per_field = {}
    for f in MERGE_FIELDS:
        use = new[f].notna()
        if f == "Type":
            cur = out.loc[idx, "Type"]
            weak = new["Type_weak"].fillna(False).astype(bool)
            use = use & (~weak | cur.isna() | (cur == "Other"))
        per_field[f] = int(use.sum())
        out.loc[idx[use.to_numpy()], f] = new.loc[use, f].values
        applied.loc[idx[use.to_numpy()]] = True

    out.loc[applied, FLAG_COL] = "scraped"
    return out, per_field


def standardize_and_merge():
    std, stats = standardize_results(RESULTS_CSV)
    flagged = pd.read_csv(INPUT_CSV)
    merged, per_field = merge_into_flagged(flagged, std)
    merged.to_csv(OUTPUT_CSV, index=False)

    print("\nStandardize + merge")
    for k, v in stats.items():
        if k != "unmapped types":
            print(f"  {k}: {v}")
    print("  values filled in, by field: " + ", ".join(f"{k} {v}" for k, v in per_field.items()))
    print("  flag counts:\n" + merged[FLAG_COL].value_counts().to_string().replace("\n", "\n    "))
    if stats["unmapped types"]:
        print("  Scraped Type labels with no mapping (Type left unchanged for these ships) -"
              " add them to SCRAPE_TYPE_MAP:")
        for k, v in stats["unmapped types"].items():
            print(f"    {k}: {v}")
    print(f"  Wrote {len(merged)} ships -> {OUTPUT_CSV}")


def main():
    flagged = pd.read_csv(INPUT_CSV)
    to_scrub = flagged.loc[flagged[FLAG_COL] == "to_scrub", MMSI_COL].astype(int).tolist()
    if not to_scrub:
        print("No ships flagged to_scrub, nothing to do.")
        return

    results = load_results(RESULTS_CSV)
    try:
        scrape(to_scrub, results)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Progress saved to {RESULTS_CSV}. Run again to resume.")
        return

    statuses = pd.Series([results[m].get("status") for m in to_scrub])
    print(f"\nScrape status:\n{statuses.value_counts().to_string()}")
    print(f"\nResults saved to {RESULTS_CSV}")

    standardize_and_merge()


if __name__ == "__main__":
    main()