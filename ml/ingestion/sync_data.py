"""Point-in-time ingestion for the AegisKEV 180-day exploitation forecast.

Sources
-------
* CISA KEV catalogue (JSON)           -> data/raw/kev/kev_catalog.parquet
* NIST NVD CVE 2.0 yearly feeds       -> data/raw/nvd/nvd_parsed.parquet
* FIRST EPSS daily archives (anchored) -> data/raw/epss_anchored.parquet

Anti-leakage contract (do not weaken):
1. CISA ``cisa*`` fields are stripped from every NVD record.
2. CVSS features come only from a metric whose ``type`` is ``Primary``.
3. EPSS scores are anchored to NVD ``published`` + 1 day (``+2`` fallback) for
   every non-rejected CVE published on/after 2021-04-14, never to KEV
   ``dateAdded`` (that would leak post-KEV EPSS spikes and drop negatives).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger("aegiskev.ingestion")

CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,19}$")

CISA_LEAK_FIELDS: tuple[str, ...] = (
    "cisaExploitAdd",
    "cisaActionDue",
    "cisaRequiredAction",
    "cisaVulnerabilityName",
)

CVSS_METRIC_ORDER: tuple[str, ...] = (
    "cvssMetricV31",
    "cvssMetricV30",
    "cvssMetricV40",
    "cvssMetricV2",
)

KEV_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/cisagov/kev-data/develop/known_exploited_vulnerabilities.json",
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
)

NVD_FEED_URL = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{year}.json.gz"
EPSS_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/empiricalsec/epss_scores/main/{year}/epss_scores-{day}.csv.gz"
)

KEV_RENAME = {
    "cveID": "cve_id",
    "vendorProject": "vendor_project",
    "vulnerabilityName": "vulnerability_name",
    "dateAdded": "date_added",
    "shortDescription": "short_description",
    "requiredAction": "required_action",
    "dueDate": "due_date",
    "knownRansomwareCampaignUse": "known_ransomware_campaign_use",
}

# Contract with ml/features/build_dataset.py. Keep these names in sync: the
# dataset builder validates every REQUIRED_* column before joining.
NVD_PARSED_COLUMNS: tuple[str, ...] = (
    "cve_id",
    "published",
    "last_modified",
    "vuln_status",
    "description_en",
    "cvss_version",
    "cvss_base_score",
    "cvss_exploitability_score",
    "cvss_impact_score",
    "is_network_av",
    "is_low_ac",
    "is_priv_none",
    "is_ui_none",
    "scope_changed",
    "cwe_primary",
    "cwe_count",
    "cpe_config_count",
    "cpe_vendor_count",
    "primary_vendor",
    "reference_count",
    "has_vendor_advisory_ref",
    "has_patch_tag",
    "has_exploit_tag",
)

EPSS_ANCHORED_COLUMNS: tuple[str, ...] = (
    "cve",
    "epss",
    "percentile",
    "anchor_date",
    "model_version",
)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested; no network, no disk)
# --------------------------------------------------------------------------- #
def is_valid_cve_id(value: Any) -> bool:
    """Return True for ``CVE-YYYY-<4..19 digits>`` with an uppercase prefix."""
    return isinstance(value, str) and CVE_PATTERN.fullmatch(value) is not None


def validate_kev_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the KEV envelope; raise ValueError on any inconsistency."""
    if not isinstance(payload, dict):
        raise ValueError("KEV payload must be a JSON object")
    vulnerabilities = payload.get("vulnerabilities")
    if not isinstance(vulnerabilities, list):
        raise ValueError("KEV payload is missing the 'vulnerabilities' list")
    declared = payload.get("count")
    if declared != len(vulnerabilities):
        raise ValueError(
            f"KEV count mismatch: payload declares {declared}, found {len(vulnerabilities)}"
        )
    invalid = [
        entry.get("cveID")
        for entry in vulnerabilities
        if not is_valid_cve_id(entry.get("cveID"))
    ]
    if invalid:
        raise ValueError(f"KEV contains invalid CVE IDs (first 5): {invalid[:5]}")
    return vulnerabilities


def strip_cisa_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Drop CISA exploitation fields so the model cannot see the target early."""
    return {key: value for key, value in record.items() if key not in CISA_LEAK_FIELDS}


def _primary_metric(metrics: dict[str, Any], version: str) -> dict[str, Any] | None:
    for entry in metrics.get(version) or []:
        if entry.get("type") == "Primary":
            return entry
    return None


def select_primary_cvss(
    metrics: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Pick the first Primary metric in v3.1 -> v3.0 -> v4.0 -> v2 order."""
    metrics = metrics or {}
    for version in CVSS_METRIC_ORDER:
        entry = _primary_metric(metrics, version)
        if entry is not None:
            return version, entry
    return None, None


def parse_vector_flags(vector: str | None) -> dict[str, bool]:
    """Map a CVSS vector string onto boolean model features."""
    flags = {
        "is_network_av": False,
        "is_low_ac": False,
        "is_priv_none": False,
        "is_ui_none": False,
        "scope_changed": False,
    }
    if not vector:
        return flags
    tokens: dict[str, str] = {}
    for part in vector.split("/"):
        key, _, value = part.partition(":")
        if value:
            tokens[key.upper()] = value.upper()
    flags["is_network_av"] = tokens.get("AV") == "N"
    flags["is_low_ac"] = tokens.get("AC") == "L"
    # CVSS v2 uses Au:N where v3+ uses PR:N.
    flags["is_priv_none"] = tokens.get("PR") == "N" or tokens.get("AU") == "N"
    flags["is_ui_none"] = tokens.get("UI") == "N"
    flags["scope_changed"] = tokens.get("S") == "C"
    return flags


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_cvss_features(metrics: dict[str, Any] | None) -> dict[str, Any]:
    version, entry = select_primary_cvss(metrics)
    features: dict[str, Any] = {
        "cvss_version": version,
        "cvss_base_score": None,
        "cvss_exploitability_score": None,
        "cvss_impact_score": None,
        **parse_vector_flags(None),
    }
    if entry is None:
        return features
    data = entry.get("cvssData") or {}
    features["cvss_base_score"] = _as_float(data.get("baseScore", entry.get("baseScore")))
    features["cvss_exploitability_score"] = _as_float(entry.get("exploitabilityScore"))
    features["cvss_impact_score"] = _as_float(entry.get("impactScore"))
    features.update(
        parse_vector_flags(data.get("vectorString") or entry.get("vectorString"))
    )
    return features


def extract_weaknesses(cve: dict[str, Any]) -> tuple[str | None, int]:
    weaknesses = cve.get("weaknesses") or []
    ids: list[str] = []
    for weakness in weaknesses:
        for description in weakness.get("description") or []:
            value = str(description.get("value") or "")
            if value.startswith("CWE-"):
                ids.append(value)
    primary: str | None = None
    for weakness in weaknesses:
        if weakness.get("type") != "Primary":
            continue
        for description in weakness.get("description") or []:
            value = str(description.get("value") or "")
            if value.startswith("CWE-"):
                primary = value
                break
        if primary is not None:
            break
    if primary is None and ids:
        primary = ids[0]
    return primary, len(dict.fromkeys(ids))


def extract_cpe_features(cve: dict[str, Any]) -> dict[str, Any]:
    configurations = cve.get("configurations") or []
    vendors: set[str] = set()
    for configuration in configurations:
        for node in configuration.get("nodes") or []:
            for match in node.get("cpeMatch") or []:
                if not match.get("vulnerable", True):
                    continue
                parts = str(match.get("criteria") or "").split(":")
                # cpe:2.3:<part>:<vendor>:<product>:...
                if len(parts) > 4:
                    vendor = parts[3].strip().lower()
                    if vendor and vendor not in {"*", "-"}:
                        vendors.add(vendor)
    return {
        "cpe_config_count": len(configurations),
        "cpe_vendor_count": len(vendors),
        "primary_vendor": sorted(vendors)[0] if vendors else "UNKNOWN",
    }


def extract_reference_features(cve: dict[str, Any]) -> dict[str, Any]:
    references = cve.get("references") or []
    tags: set[str] = set()
    for reference in references:
        tags.update(reference.get("tags") or [])
    return {
        "reference_count": len(references),
        "has_vendor_advisory_ref": "Vendor Advisory" in tags,
        "has_patch_tag": "Patch" in tags,
        "has_exploit_tag": "Exploit" in tags,
    }


def extract_english_description(cve: dict[str, Any]) -> str:
    for description in cve.get("descriptions") or []:
        if description.get("lang") == "en":
            return str(description.get("value") or "")
    return ""


def is_rejected(item: dict[str, Any]) -> bool:
    status = str((item.get("cve") or {}).get("vulnStatus") or "").strip().lower()
    return status in {"rejected", "reject"}


def build_nvd_record(item: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten one NVD ``vulnerabilities[]`` entry; None for invalid/absent IDs."""
    cve = strip_cisa_fields(item.get("cve") or {})
    cve_id = cve.get("id")
    if not is_valid_cve_id(cve_id):
        return None
    record: dict[str, Any] = {
        "cve_id": cve_id,
        "published": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "vuln_status": cve.get("vulnStatus"),
        "description_en": extract_english_description(cve),
    }
    record.update(extract_cvss_features(cve.get("metrics") or {}))
    primary_cwe, cwe_count = extract_weaknesses(cve)
    record["cwe_primary"] = primary_cwe
    record["cwe_count"] = cwe_count
    record.update(extract_cpe_features(cve))
    record.update(extract_reference_features(cve))
    return record


def decode_gzip_text(raw: bytes) -> str:
    """Gunzip (possibly nested) bytes, then decode UTF-8 with replacement.

    EPSS archives are ``.csv.gz`` both from the HTTP response and from the
    on-disk cache. Decoding those bytes directly yields
    ``'utf-8' codec can't decode byte 0x8b`` and zero anchored rows.
    """
    while raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def parse_epss_payload(raw: bytes | str) -> pd.DataFrame:
    """Parse an EPSS daily archive.

    Handles v1 files (no header, data starts immediately) and v2+ files that
    begin with ``#model_version:...`` comment lines followed by a CSV header.
    Bytes may be gzip-compressed (the normal archive format).
    """
    text = decode_gzip_text(raw) if isinstance(raw, bytes) else raw
    model_version: str | None = None
    data_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "model_version" in line and ":" in line:
                model_version = line.split(":", 1)[1].strip() or None
            continue
        data_lines.append(line)

    columns = ["cve", "epss", "percentile", "model_version"]
    if not data_lines:
        return pd.DataFrame(columns=columns)

    first = data_lines[0].lower().replace(" ", "")
    rows = data_lines[1:] if first.startswith("cve,") else data_lines

    records: list[dict[str, Any]] = []
    for line in rows:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not is_valid_cve_id(parts[0]):
            continue
        try:
            epss = float(parts[1])
            percentile = float(parts[2])
        except ValueError:
            continue
        records.append(
            {
                "cve": parts[0],
                "epss": epss,
                "percentile": percentile,
                "model_version": model_version,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


# --------------------------------------------------------------------------- #
# Network + disk
# --------------------------------------------------------------------------- #
def build_session(retries: int = 5) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "AegisKEV/0.1 (data ingestion)"})
    return session


def fetch_kev_catalog(
    session: requests.Session | None = None, timeout: int = 60
) -> list[dict[str, Any]]:
    """Download and validate the KEV catalogue, falling back to cisa.gov."""
    session = session or build_session()
    errors: list[str] = []
    for url in KEV_URLS:
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return validate_kev_payload(response.json())
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {exc}")
            LOGGER.warning("KEV fetch failed from %s: %s", url, exc)
    raise RuntimeError("Unable to fetch CISA KEV catalogue: " + "; ".join(errors))


def kev_frame(vulnerabilities: Sequence[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(vulnerabilities))
    frame = frame.rename(
        columns={src: dst for src, dst in KEV_RENAME.items() if src in frame.columns}
    )
    for column in ("date_added", "due_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def sync_kev(
    raw_dir: Path, session: requests.Session | None = None, timeout: int = 60
) -> pd.DataFrame:
    frame = kev_frame(fetch_kev_catalog(session, timeout))
    out = Path(raw_dir) / "kev" / "kev_catalog.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    LOGGER.info("Saved %s KEV rows -> %s", len(frame), out)
    return frame


def _nvd_cache_path(raw_dir: Path, year: int) -> Path:
    return Path(raw_dir) / "nvd" / f"nvdcve-2.0-{year}.json.gz"


def download_nvd_year(
    year: int,
    raw_dir: Path,
    session: requests.Session | None = None,
    force: bool = False,
    timeout: int = 180,
) -> list[dict[str, Any]]:
    """Download (or reuse) one yearly NVD feed and return its vulnerabilities."""
    session = session or build_session()
    cache = _nvd_cache_path(raw_dir, year)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or force:
        url = NVD_FEED_URL.format(year=year)
        LOGGER.info("Downloading NVD feed %s", url)
        response = session.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        tmp = cache.with_suffix(cache.suffix + ".part")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
        tmp.replace(cache)
    with gzip.open(cache, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("vulnerabilities") or []


def sync_nvd(
    years: Sequence[int],
    raw_dir: Path,
    session: requests.Session | None = None,
    force: bool = False,
    workers: int = 4,
) -> pd.DataFrame:
    session = session or build_session()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(download_nvd_year, year, Path(raw_dir), session, force): year
            for year in years
        }
        for future in as_completed(futures):
            year = futures[future]
            items = future.result()
            kept = 0
            for item in items:
                if is_rejected(item):
                    continue
                record = build_nvd_record(item)
                if record is None:
                    continue
                records.append(record)
                kept += 1
            LOGGER.info("NVD %s: kept %s/%s CVEs", year, kept, len(items))
    frame = pd.DataFrame.from_records(records, columns=NVD_PARSED_COLUMNS)
    out = Path(raw_dir) / "nvd" / "nvd_parsed.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    LOGGER.info("Saved %s NVD rows -> %s", len(frame), out)
    return frame


def _epss_cache_path(cache_dir: Path, day: date) -> Path:
    return Path(cache_dir) / f"{day.year}" / f"epss_scores-{day.isoformat()}.csv.gz"


def download_epss_archive(
    day: date,
    cache_dir: Path,
    session: requests.Session | None = None,
    force: bool = False,
    timeout: int = 60,
) -> pd.DataFrame | None:
    """Fetch one daily EPSS archive (cached), or None when it does not exist."""
    session = session or build_session()
    path = _epss_cache_path(cache_dir, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or force:
        url = EPSS_ARCHIVE_URL.format(year=day.year, day=day.isoformat())
        response = session.get(url, timeout=timeout)
        if response.status_code == 404:
            LOGGER.warning("EPSS archive missing for %s", day)
            return None
        response.raise_for_status()
        path.write_bytes(response.content)
    return parse_epss_payload(path.read_bytes())


def _fetch_epss_days(
    days: Sequence[date],
    cache_dir: Path,
    workers: int,
    session: requests.Session,
    force: bool,
) -> dict[date, pd.DataFrame | None]:
    results: dict[date, pd.DataFrame | None] = {}
    if not days:
        return results
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(download_epss_archive, day, cache_dir, session, force): day
            for day in days
        }
        for future in as_completed(futures):
            day = futures[future]
            try:
                results[day] = future.result()
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                LOGGER.warning("EPSS fetch failed for %s: %s", day, exc)
                results[day] = None
    return results


def _collect_epss_rows(
    frame: pd.DataFrame | None,
    cves: set[str],
    anchor: date,
    rows: list[dict[str, Any]],
) -> set[str]:
    if frame is None or frame.empty:
        return set()
    subset = frame[frame["cve"].isin(cves)]
    for record in subset.to_dict("records"):
        rows.append(
            {
                "cve": record["cve"],
                "epss": record["epss"],
                "percentile": record["percentile"],
                "anchor_date": anchor.isoformat(),
                "model_version": record.get("model_version"),
            }
        )
    return set(subset["cve"])


def fetch_epss_anchored(
    nvd_df: pd.DataFrame,
    raw_dir: Path,
    workers: int = 8,
    session: requests.Session | None = None,
    force: bool = False,
    start_date: date = date(2021, 4, 14),
) -> pd.DataFrame:
    """Fetch EPSS for every NVD CVE at ``published + 1 day`` (``+2`` fallback).

    The anchor is the NVD publication date, *never* the KEV ``dateAdded``:
    anchoring on KEV would (a) leak EPSS spikes that happen after disclosure,
    and (b) silently drop every negative CVE that never enters KEV.
    """
    if "published" not in nvd_df.columns:
        raise ValueError(
            "EPSS anchoring requires NVD 'published' dates; "
            "KEV date_added is not a valid anchor"
        )
    if "cve_id" not in nvd_df.columns:
        raise ValueError("EPSS anchoring requires a 'cve_id' column")

    session = session or build_session()
    cache_dir = Path(raw_dir) / "epss_cache"

    published = pd.to_datetime(nvd_df["published"], errors="coerce")
    cve_ids = nvd_df["cve_id"].astype(str)
    eligible = published.notna() & (published >= pd.Timestamp(start_date))

    # One anchor per CVE (earliest publication wins for duplicate NVD rows).
    anchors: dict[str, date] = {}
    for cve_id, published_at in zip(cve_ids[eligible], published[eligible]):
        anchor = published_at.date() + timedelta(days=1)
        if cve_id not in anchors or anchor < anchors[cve_id]:
            anchors[cve_id] = anchor

    groups: dict[date, set[str]] = {}
    for cve_id, anchor in anchors.items():
        groups.setdefault(anchor, set()).add(cve_id)

    frames = _fetch_epss_days(sorted(groups), cache_dir, workers, session, force)
    rows: list[dict[str, Any]] = []
    fallback: dict[date, set[str]] = {}
    for anchor, cves in groups.items():
        found = _collect_epss_rows(frames.get(anchor), cves, anchor, rows)
        missing = cves - found
        if missing:
            fallback.setdefault(anchor + timedelta(days=1), set()).update(missing)

    if fallback:
        extra_days = sorted(set(fallback) - set(frames))
        frames.update(_fetch_epss_days(extra_days, cache_dir, workers, session, force))
        for day, cves in fallback.items():
            _collect_epss_rows(frames.get(day), cves, day, rows)

    return pd.DataFrame.from_records(rows, columns=EPSS_ANCHORED_COLUMNS)


def sync_epss(
    nvd_df: pd.DataFrame,
    raw_dir: Path,
    workers: int = 8,
    session: requests.Session | None = None,
    force: bool = False,
) -> pd.DataFrame:
    frame = fetch_epss_anchored(nvd_df, raw_dir, workers, session, force)
    out = Path(raw_dir) / "epss_anchored.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    LOGGER.info("Saved %s anchored EPSS rows -> %s", len(frame), out)
    return frame


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def expand_years(values: Sequence[str]) -> list[int]:
    """Accept ``2021-2026`` ranges and/or explicit years; return sorted uniques."""
    years: list[int] = []
    for value in values:
        text = str(value).strip()
        if "-" in text:
            start, end = text.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(text))
    return sorted(dict.fromkeys(years))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest point-in-time KEV, NVD and EPSS data for AegisKEV."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2021-2026"],
        help="NVD years, e.g. 2021-2026 or 2021 2022 (default: 2021-2026)",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw"), help="raw data root"
    )
    parser.add_argument("--workers", type=int, default=8, help="EPSS download threads")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=("kev", "nvd", "epss"),
        default=["kev", "nvd", "epss"],
        help="which ingestion steps to run (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download cached archives"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    years = expand_years(args.years)
    session = build_session()

    kev_df: pd.DataFrame | None = None
    if "kev" in args.steps:
        kev_df = sync_kev(raw_dir, session)

    nvd_df: pd.DataFrame | None = None
    if "nvd" in args.steps:
        nvd_df = sync_nvd(years, raw_dir, session, force=args.force)

    if "epss" in args.steps:
        if nvd_df is None:
            nvd_path = raw_dir / "nvd" / "nvd_parsed.parquet"
            if not nvd_path.exists():
                raise FileNotFoundError(
                    f"{nvd_path} is required for EPSS anchoring; run --steps nvd first"
                )
            nvd_df = pd.read_parquet(nvd_path)
        sync_epss(nvd_df, raw_dir, args.workers, session, force=args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
