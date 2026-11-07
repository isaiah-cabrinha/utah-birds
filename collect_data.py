"""
collect_data.py
===============

Builds the Utah birds dataset by combining two sources:

1. Research-grade bird observations in Utah from the iNaturalist API (v1).
2. Species-level traits (body mass, migration, habitat, diet) from AVONET
   (Tobias et al. 2022, Ecology Letters), which is licensed CC BY 4.0.

Before running
--------------
* Download "AVONET Supplementary dataset 1.xlsx" from the AVONET figshare page
  and save it to data/AVONET Supplementary dataset 1.xlsx.
* Put your own email in the HEADERS User-Agent below.
* Install packages: uv add requests pandas openpyxl

Run
---
    uv run python collect_data.py

Outputs
-------
data/raw/                   cached API responses, one JSON file per page (not committed)
data/utah_birds_full.csv    every observation (not committed, mixed licenses)
data/utah_birds.csv         only CC0 / CC BY observations (safe to publish)
data/unmatched_species.csv  iNaturalist species with no AVONET match (for hand-checking)

Good-citizen practices
----------------------
* At most one request per second (iNaturalist asks for 60/minute or fewer).
* Every page is cached, so re-running the script never re-downloads data.
* A User-Agent identifies the project.
* No usernames, user IDs, photos, or free-text location descriptions are kept.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BASE_URL = "https://api.inaturalist.org/v1/observations"
HEADERS = {"User-Agent": "STAT386 data acquisition project (cabrinhi@byu.edu)"}

UTAH_PLACE_ID = 52        # from /v1/places/autocomplete?q=Utah -> "Utah, US"
YEAR = 2025
PER_PAGE = 200            # the API maximum
MAX_RESULTS = 10_000      # the API will not page past 10,000 results per query
PAUSE_SECONDS = 1.0       # pause after every live request

SHAREABLE_LICENSES = {"cc0", "cc-by"}

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
AVONET_PATH = DATA_DIR / "AVONET Supplementary dataset 1.xlsx"
AVONET_SHEET = "AVONET2_eBird"  # eBird taxonomy, closest to iNaturalist's bird names

# iNaturalist follows newer taxonomy than AVONET (2022). These species were
# renamed or split since then, so map each iNaturalist name to the AVONET
# name(s) that cover it. The first candidate found in AVONET is used.
NAME_FIXES = {
    "Astur cooperii": ["Accipiter cooperii"],                 # Cooper's Hawk
    "Astur atricapillus": ["Accipiter atricapillus", "Accipiter gentilis"],  # Goshawk (split)
    "Setophaga aestiva": ["Setophaga petechia"],              # Yellow Warbler (split)
    "Leuconotopicus villosus": ["Dryobates villosus", "Picoides villosus"],  # Hairy Woodpecker
    "Tyto furcata": ["Tyto alba"],                            # Barn Owl (split)
    "Cistothorus paludicola": ["Cistothorus palustris"],      # Marsh Wren (split)
    "Ardea ibis": ["Bubulcus ibis"],                          # Cattle Egret
    "Vireo swainsoni": ["Vireo gilvus"],                      # Warbling Vireo (split)
    "Anarhynchus nivosus": ["Charadrius nivosus"],            # Snowy Plover
    "Larus smithsonianus": ["Larus argentatus"],              # Herring Gull (split)
    "Nannus pacificus": ["Troglodytes pacificus", "Nannus pacificus"],  # Pacific Wren
    "Numenius hudsonicus": ["Numenius phaeopus"],             # Whimbrel (split)
    "Hesperiphona vespertina": ["Coccothraustes vespertinus"],  # Evening Grosbeak
}

MIGRATION_LABELS = {1: "sedentary", 2: "partial migrant", 3: "migratory"}
SEASONS = {12: "winter", 1: "winter", 2: "winter",
           3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer",
           9: "fall", 10: "fall", 11: "fall"}


# ---------------------------------------------------------------------------
# Step 1: download observations from iNaturalist
# ---------------------------------------------------------------------------
def fetch_page(year: int, month: int, page: int) -> dict:
    """Return one page of observations, reading from the cache when possible."""
    cache_file = RAW_DIR / f"inat_{year}_{month:02d}_p{page:02d}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    last_day = pd.Period(f"{year}-{month:02d}").days_in_month
    params = {
        "place_id": UTAH_PLACE_ID,
        "iconic_taxa": "Aves",
        "quality_grade": "research",
        "d1": f"{year}-{month:02d}-01",
        "d2": f"{year}-{month:02d}-{last_day}",
        "per_page": PER_PAGE,
        "page": page,
        "order_by": "id",
        "order": "asc",
    }
    data = get_with_retries(params)

    cache_file.write_text(json.dumps(data))
    time.sleep(PAUSE_SECONDS)
    return data


def get_with_retries(params: dict, max_attempts: int = 5) -> dict:
    """GET one page, retrying dropped connections and server errors.

    Waits 5, 10, 20, 40 seconds between attempts so a hiccup doesn't crash
    the whole run. If the API sends a 429 (too many requests), it waits as
    long as the Retry-After header asks.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                wait = int(response.headers.get("Retry-After", 5 * 2 ** (attempt - 1)))
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as err:
            if attempt == max_attempts:
                raise
            if not isinstance(err, requests.HTTPError) or err.response is None:
                wait = 5 * 2 ** (attempt - 1)
            elif err.response.status_code < 500 and err.response.status_code != 429:
                raise  # a 4xx like 400/404 won't fix itself, so stop
            print(f"    {type(err).__name__} on page {params['page']}; "
                  f"retrying in {wait}s (attempt {attempt}/{max_attempts})")
            time.sleep(wait)


def fetch_month(year: int, month: int) -> list[dict]:
    """Download every observation for one month, page by page."""
    first_page = fetch_page(year, month, 1)
    total = first_page["total_results"]
    if total > MAX_RESULTS:
        print(f"  Warning: {total:,} results in {year}-{month:02d}; "
              f"only the first {MAX_RESULTS:,} can be paged. Split the month in half.")

    n_pages = min(math.ceil(total / PER_PAGE), MAX_RESULTS // PER_PAGE)
    results = list(first_page["results"])
    for page in range(2, n_pages + 1):
        results.extend(fetch_page(year, month, page)["results"])

    print(f"  {year}-{month:02d}: {len(results):,} observations")
    return results


# ---------------------------------------------------------------------------
# Step 2: flatten the JSON into one row per observation
# ---------------------------------------------------------------------------
def to_row(obs: dict) -> dict:
    """Keep only the fields we need. User information is deliberately dropped."""
    taxon = obs.get("taxon") or {}
    location = obs.get("location")
    lat, lon = map(float, location.split(",")) if location else (None, None)
    return {
        "obs_id": obs.get("id"),
        "observed_on": obs.get("observed_on"),
        "latitude": lat,
        "longitude": lon,
        "positional_accuracy_m": obs.get("positional_accuracy"),
        "coords_obscured": obs.get("obscured"),
        "common_name": taxon.get("preferred_common_name"),
        "inat_name": taxon.get("name"),
        "taxon_rank": taxon.get("rank"),
        "id_agreements": obs.get("num_identification_agreements"),
        "license": obs.get("license_code"),
    }


def build_observations(year: int) -> pd.DataFrame:
    """Download a full year and return a tidy observation table."""
    records = []
    for month in range(1, 13):
        records.extend(fetch_month(year, month))

    df = pd.DataFrame([to_row(obs) for obs in records]).drop_duplicates("obs_id")

    # Keep species-level IDs; collapse subspecies to their species name
    df = df[df["taxon_rank"].isin(["species", "subspecies"])].copy()
    df["scientific_name"] = df["inat_name"].str.split().str[:2].str.join(" ")

    # Date features
    df["observed_on"] = pd.to_datetime(df["observed_on"])
    df["month"] = df["observed_on"].dt.month
    df["day_of_year"] = df["observed_on"].dt.dayofyear
    df["season"] = df["month"].map(SEASONS)
    return df.drop(columns=["inat_name", "taxon_rank"])


# ---------------------------------------------------------------------------
# Step 3: add species traits from AVONET
# ---------------------------------------------------------------------------
def load_traits() -> pd.DataFrame:
    """Read the AVONET eBird-taxonomy sheet and keep a few useful traits."""
    traits = pd.read_excel(AVONET_PATH, sheet_name=AVONET_SHEET)
    columns = {
        "Species2": "scientific_name",
        "Family2": "family",
        "Order2": "order",
        "Mass": "body_mass_g",
        "Migration": "migration_code",
        "Habitat": "habitat",
        "Trophic.Level": "trophic_level",
    }
    missing = set(columns) - set(traits.columns)
    if missing:
        raise KeyError(f"AVONET sheet is missing columns {missing}; check the sheet name.")

    traits = traits[list(columns)].rename(columns=columns)
    traits["migration"] = traits["migration_code"].map(MIGRATION_LABELS)
    return traits.drop(columns="migration_code")


def match_avonet_name(name: str, avonet_names: set[str]) -> str:
    """Return the AVONET name for an iNaturalist species name.

    Uses the name as-is when AVONET has it; otherwise tries each candidate in
    NAME_FIXES. The scientific_name column keeps iNaturalist's current name.
    """
    if name in avonet_names:
        return name
    for candidate in NAME_FIXES.get(name, []):
        if candidate in avonet_names:
            return candidate
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading iNaturalist observations...")
    obs = build_observations(YEAR)

    print("Merging AVONET traits...")
    traits = load_traits()
    obs["avonet_name"] = obs["scientific_name"].map(
        lambda name: match_avonet_name(name, set(traits["scientific_name"])))
    birds = (obs.merge(traits.rename(columns={"scientific_name": "avonet_name"}),
                       on="avonet_name", how="left")
                .drop(columns="avonet_name"))

    unmatched = (birds.loc[birds["body_mass_g"].isna(), ["scientific_name", "common_name"]]
                 .value_counts().reset_index(name="n_obs"))
    unmatched.to_csv(DATA_DIR / "unmatched_species.csv", index=False)

    birds.to_csv(DATA_DIR / "utah_birds_full.csv", index=False)
    shareable = birds[birds["license"].isin(SHAREABLE_LICENSES)]
    shareable.to_csv(DATA_DIR / "utah_birds.csv", index=False)

    print(f"\nDone: {len(birds):,} observations of {birds['scientific_name'].nunique()} species")
    print(f"  {len(shareable):,} are CC0/CC BY and saved to data/utah_birds.csv")
    print(f"  {len(unmatched)} species had no AVONET match (see data/unmatched_species.csv)")


if __name__ == "__main__":
    main()