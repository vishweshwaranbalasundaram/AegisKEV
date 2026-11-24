"""Unit tests for point-in-time ingestion (no network, no disk)."""

from __future__ import annotations

import gzip
from datetime import date

import pandas as pd
import pytest

from ml.ingestion import sync_data
from ml.ingestion.sync_data import (
    CISA_LEAK_FIELDS,
    build_nvd_record,
    decode_gzip_text,
    expand_years,
    extract_cpe_features,
    is_rejected,
    is_valid_cve_id,
    parse_epss_payload,
    parse_vector_flags,
    select_primary_cvss,
    strip_cisa_fields,
    validate_kev_payload,
)


def _metric(score, metric_type="Primary", vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"):
    return {
        "type": metric_type,
        "cvssData": {"baseScore": score, "vectorString": vector},
        "exploitabilityScore": 3.9,
        "impactScore": 5.9,
    }


# --------------------------------------------------------------------------- #
# CVE identifier regex
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cve_id",
    [
        "CVE-2021-44228",  # 5-digit sequence
        "CVE-2024-1234",  # minimum 4 digits
        "CVE-1999-1234567890123456789",  # maximum 19 digits
    ],
)
def test_cve_id_regex_accepts_valid(cve_id):
    assert is_valid_cve_id(cve_id)


@pytest.mark.parametrize(
    "cve_id",
    [
        "CVE-2021-123",  # 3 digits: too short
        "CVE-2021-12345678901234567890",  # 20 digits: too long
        "cve-2021-44228",  # lowercase prefix
        "CVE-21-44228",  # 2-digit year
        "CVE-2021-4422a",  # non-digit
        "CVE-2021-44228 ",  # trailing space
        "GHSA-xxxx-yyyy-zzzz",
        "",
        None,
    ],
)
def test_cve_id_regex_rejects_invalid(cve_id):
    assert not is_valid_cve_id(cve_id)


# --------------------------------------------------------------------------- #
# KEV envelope validation
# --------------------------------------------------------------------------- #
def test_kev_payload_count_mismatch_raises():
    payload = {"count": 3, "vulnerabilities": [{"cveID": "CVE-2021-44228"}]}
    with pytest.raises(ValueError, match="count mismatch"):
        validate_kev_payload(payload)


def test_kev_payload_invalid_cve_raises():
    payload = {"count": 1, "vulnerabilities": [{"cveID": "CVE-2021-123"}]}
    with pytest.raises(ValueError, match="invalid CVE"):
        validate_kev_payload(payload)


def test_kev_payload_accepts_consistent_envelope():
    payload = {"count": 1, "vulnerabilities": [{"cveID": "CVE-2021-44228"}]}
    assert validate_kev_payload(payload) == payload["vulnerabilities"]


# --------------------------------------------------------------------------- #
# CVSS: Primary only, never index 0
# --------------------------------------------------------------------------- #
def test_primary_cvss_wins_over_secondary_at_index_zero():
    metrics = {
        "cvssMetricV31": [
            _metric(9.8, metric_type="Secondary"),
            _metric(5.3, metric_type="Primary"),
        ]
    }
    version, entry = select_primary_cvss(metrics)
    assert version == "cvssMetricV31"
    assert entry["cvssData"]["baseScore"] == 5.3


def test_secondary_only_version_is_skipped_for_lower_version_primary():
    metrics = {
        "cvssMetricV31": [_metric(9.8, metric_type="Secondary")],
        "cvssMetricV30": [_metric(7.5, metric_type="Primary")],
    }
    version, entry = select_primary_cvss(metrics)
    assert version == "cvssMetricV30"
    assert entry["cvssData"]["baseScore"] == 7.5


def test_no_primary_metric_yields_no_cvss():
    metrics = {"cvssMetricV31": [_metric(9.8, metric_type="Secondary")]}
    assert select_primary_cvss(metrics) == (None, None)


def test_build_nvd_record_uses_primary_cvss_score():
    item = {
        "cve": {
            "id": "CVE-2021-44228",
            "metrics": {
                "cvssMetricV31": [
                    _metric(9.8, metric_type="Secondary"),
                    _metric(10.0, metric_type="Primary"),
                ]
            },
        }
    }
    record = build_nvd_record(item)
    assert record["cvss_base_score"] == 10.0
    assert record["cvss_version"] == "cvssMetricV31"
    assert record["is_network_av"] is True
    assert record["is_low_ac"] is True
    assert record["is_priv_none"] is True
    assert record["is_ui_none"] is True
    assert record["scope_changed"] is False


def test_vector_flags_scope_changed_and_cvss_v2_privileges():
    flags = parse_vector_flags("AV:N/AC:L/Au:N/C:P/I:P/A:P")
    assert flags["is_priv_none"] is True
    assert parse_vector_flags("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H")[
        "scope_changed"
    ] is True
    assert parse_vector_flags(None)["is_network_av"] is False


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #
def _leaky_item():
    return {
        "cve": {
            "id": "CVE-2021-44228",
            "published": "2021-12-10T10:15:09.143",
            "lastModified": "2024-11-21T01:00:00.000",
            "vulnStatus": "Analyzed",
            "cisaExploitAdd": "2021-12-10",
            "cisaActionDue": "2021-12-24",
            "cisaRequiredAction": "Apply updates per vendor instructions.",
            "cisaVulnerabilityName": "Apache Log4j2 Remote Code Execution",
            "descriptions": [{"lang": "en", "value": "JNDI injection."}],
            "metrics": {"cvssMetricV31": [_metric(10.0)]},
        }
    }


def test_strip_cisa_fields_removes_all_four_and_does_not_mutate():
    source = _leaky_item()["cve"]
    stripped = strip_cisa_fields(source)
    for field in CISA_LEAK_FIELDS:
        assert field not in stripped
        assert field in source  # original untouched


def test_build_nvd_record_never_exposes_cisa_fields():
    record = build_nvd_record(_leaky_item())
    for field in CISA_LEAK_FIELDS:
        assert field not in record
    assert record["cve_id"] == "CVE-2021-44228"
    assert record["description_en"] == "JNDI injection."


def _full_nvd_item():
    item = _leaky_item()
    item["cve"]["weaknesses"] = [
        {"type": "Primary", "description": [{"lang": "en", "value": "CWE-502"}]}
    ]
    item["cve"]["configurations"] = [
        {
            "nodes": [
                {
                    "cpeMatch": [
                        {
                            "vulnerable": True,
                            "criteria": "cpe:2.3:a:apache:log4j:2.0:*:*:*:*:*:*:*",
                        }
                    ]
                }
            ]
        }
    ]
    item["cve"]["references"] = [
        {"url": "https://example.test/advisory", "tags": ["Vendor Advisory"]},
        {"url": "https://example.test/exploit", "tags": ["Exploit"]},
    ]
    return item


def test_nvd_parsed_schema_matches_build_dataset_contract():
    """nvd_parsed column names must match what build_dataset.py requires."""
    from ml.features.build_dataset import REQUIRED_NVD_COLUMNS

    record = build_nvd_record(_full_nvd_item())
    assert set(record) == set(sync_data.NVD_PARSED_COLUMNS)
    assert set(REQUIRED_NVD_COLUMNS) <= set(record)
    assert record["cwe_primary"] == "CWE-502"
    assert record["description_en"] == "JNDI injection."
    assert record["primary_vendor"] == "apache"


def test_epss_anchored_schema_matches_build_dataset_contract():
    from ml.features.build_dataset import REQUIRED_EPSS_COLUMNS

    assert set(REQUIRED_EPSS_COLUMNS) <= set(sync_data.EPSS_ANCHORED_COLUMNS)


def test_rejected_cves_are_flagged_for_dropping():
    assert is_rejected({"cve": {"vulnStatus": "Rejected"}}) is True
    assert is_rejected({"cve": {"vulnStatus": "REJECT"}}) is True
    assert is_rejected({"cve": {"vulnStatus": "Analyzed"}}) is False


# --------------------------------------------------------------------------- #
# CPE aggregation
# --------------------------------------------------------------------------- #
def test_cpe_features_pick_sorted_vendor_and_count():
    cve = {
        "configurations": [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {"vulnerable": True, "criteria": "cpe:2.3:a:zebra:z:1:*:*:*:*:*:*:*"},
                            {"vulnerable": True, "criteria": "cpe:2.3:a:apache:log4j:2:*:*:*:*:*:*:*"},
                            {"vulnerable": False, "criteria": "cpe:2.3:a:ignored:ignored:1:*:*:*:*:*:*:*"},
                        ]
                    }
                ]
            }
        ]
    }
    features = extract_cpe_features(cve)
    assert features["cpe_config_count"] == 1
    assert features["cpe_vendor_count"] == 2
    assert features["primary_vendor"] == "apache"
    assert extract_cpe_features({})["primary_vendor"] == "UNKNOWN"


# --------------------------------------------------------------------------- #
# EPSS archive parsing
# --------------------------------------------------------------------------- #
def test_parse_epss_v1_headerless():
    raw = (
        "CVE-2021-44228,0.97000,0.99999\n"
        "CVE-2022-22965,0.50000,0.90000\n"
    )
    frame = parse_epss_payload(raw)
    assert list(frame.columns) == ["cve", "epss", "percentile", "model_version"]
    assert len(frame) == 2
    assert frame.iloc[0]["cve"] == "CVE-2021-44228"
    assert frame.iloc[0]["epss"] == pytest.approx(0.97)
    assert frame.iloc[0]["model_version"] is None


def test_parse_epss_v4_model_version_header():
    raw = (
        "#model_version:v2025.03.14\n"
        "#score_date:2025-03-14T00:00:00Z\n"
        "cve,epss,percentile\n"
        "CVE-2021-44228,0.94,0.99\n"
        "CVE-2023-0001,0.01,0.10\n"
    )
    frame = parse_epss_payload(raw)
    assert len(frame) == 2
    assert frame.iloc[0]["model_version"] == "v2025.03.14"
    assert frame.iloc[1]["epss"] == pytest.approx(0.01)


def test_parse_epss_skips_invalid_rows_and_empty_payload():
    frame = parse_epss_payload("cve,epss,percentile\nNOT-A-CVE,0.1,0.2\nCVE-2024-1234,oops,0.2\n")
    assert frame.empty
    empty = parse_epss_payload("#model_version:v4\n")
    assert empty.empty
    assert list(empty.columns) == ["cve", "epss", "percentile", "model_version"]


EPSS_GZ_PAYLOAD = (
    b"#model_version:v2025.03.14\n"
    b"#score_date:2025-03-14T00:00:00Z\n"
    b"cve,epss,percentile\n"
    b"CVE-2021-44228,0.94,0.99\n"
    b"CVE-2023-0001,0.01,0.10\n"
)


def test_decode_gzip_text_handles_nested_archives_and_bad_bytes():
    nested = gzip.compress(gzip.compress(EPSS_GZ_PAYLOAD))
    assert decode_gzip_text(nested).startswith("#model_version:v2025.03.14")
    assert decode_gzip_text(b"\xff") == "\ufffd"  # replacement, never raises


def test_download_epss_archive_gunzips_cached_csv_gz(tmp_path):
    """Regression: cached .csv.gz bytes must be gunzipped, not decoded as UTF-8."""
    day = date(2021, 12, 11)
    cache_dir = tmp_path / "epss_cache"
    cache = sync_data._epss_cache_path(cache_dir, day)
    cache.parent.mkdir(parents=True)
    cache.write_bytes(gzip.compress(EPSS_GZ_PAYLOAD))

    class NoNetworkSession:
        def get(self, *args, **kwargs):
            raise AssertionError("a cached archive must not be re-downloaded")

    frame = sync_data.download_epss_archive(
        day, cache_dir, session=NoNetworkSession()
    )

    assert len(frame) == 2
    assert frame.iloc[0]["cve"] == "CVE-2021-44228"
    assert frame.iloc[0]["epss"] == pytest.approx(0.94)
    assert frame.iloc[0]["model_version"] == "v2025.03.14"


def test_download_epss_archive_gunzips_http_response_bytes(tmp_path):
    """Regression: gzip bytes straight from HTTP must parse without a cache hit."""
    day = date(2021, 12, 11)

    class FakeResponse:
        status_code = 200
        content = gzip.compress(EPSS_GZ_PAYLOAD)

        def raise_for_status(self):
            return None

    class FakeSession:
        def get(self, url, timeout):
            assert url.endswith("epss_scores-2021-12-11.csv.gz")
            return FakeResponse()

    frame = sync_data.download_epss_archive(
        day, tmp_path / "epss_cache", session=FakeSession()
    )

    assert len(frame) == 2
    assert frame.iloc[1]["cve"] == "CVE-2023-0001"
    assert frame.iloc[1]["epss"] == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def test_expand_years_handles_ranges_and_explicit_years():
    assert expand_years(["2021-2023", "2023", "2026"]) == [2021, 2022, 2023, 2026]


# --------------------------------------------------------------------------- #
# EPSS anchoring: NVD published + 1 day, never KEV dateAdded
# --------------------------------------------------------------------------- #
def _epss_frame(rows):
    return pd.DataFrame(rows, columns=["cve", "epss", "percentile"]).assign(
        model_version="v4"
    )


def test_epss_anchors_on_nvd_published_plus_one(monkeypatch, tmp_path):
    nvd = pd.DataFrame(
        {
            "cve_id": ["CVE-2021-44228", "CVE-2022-22965"],
            "published": ["2021-12-10T10:15:09", "2022-04-01T00:00:00"],
        }
    )
    frames = {
        date(2021, 12, 11): _epss_frame([("CVE-2021-44228", 0.9, 0.99)]),
        date(2022, 4, 2): None,  # +1 archive missing -> +2 fallback
        date(2022, 4, 3): _epss_frame([("CVE-2022-22965", 0.4, 0.8)]),
    }
    requested: list[date] = []

    def fake_fetch(days, cache_dir, workers, session, force):
        requested.extend(days)
        return {day: frames.get(day) for day in days}

    monkeypatch.setattr(sync_data, "_fetch_epss_days", fake_fetch)
    out = sync_data.fetch_epss_anchored(nvd, tmp_path, session=object())

    anchors = dict(zip(out["cve"], out["anchor_date"]))
    assert anchors["CVE-2021-44228"] == "2021-12-11"
    assert anchors["CVE-2022-22965"] == "2022-04-03"
    assert date(2022, 4, 2) in requested and date(2022, 4, 3) in requested


def test_epss_anchor_ignores_date_added_and_old_publications(monkeypatch, tmp_path):
    nvd = pd.DataFrame(
        {
            "cve_id": ["CVE-2020-0001", "CVE-2021-0002"],
            "published": ["2020-06-01", None],
            # Wildly different KEV dates must have no effect on the anchor.
            "date_added": ["2024-01-01", "2024-01-01"],
        }
    )
    requested: list[date] = []

    def fake_fetch(days, cache_dir, workers, session, force):
        requested.extend(days)
        return {}

    monkeypatch.setattr(sync_data, "_fetch_epss_days", fake_fetch)
    out = sync_data.fetch_epss_anchored(nvd, tmp_path, session=object())

    assert out.empty
    assert requested == []  # pre-2021-04-14 and missing published dates are skipped


def test_epss_anchor_start_date_is_inclusive(monkeypatch, tmp_path):
    nvd = pd.DataFrame(
        {"cve_id": ["CVE-2021-1111"], "published": ["2021-04-14T12:00:00"]}
    )
    seen: list[date] = []

    def fake_fetch(days, cache_dir, workers, session, force):
        seen.extend(days)
        return {
            day: _epss_frame([("CVE-2021-1111", 0.2, 0.5)]) for day in days
        }

    monkeypatch.setattr(sync_data, "_fetch_epss_days", fake_fetch)
    out = sync_data.fetch_epss_anchored(nvd, tmp_path, session=object())

    assert seen == [date(2021, 4, 15)]
    assert out.iloc[0]["anchor_date"] == "2021-04-15"


def test_epss_anchor_rejects_kev_only_frame(tmp_path):
    kev_only = pd.DataFrame(
        {"cve_id": ["CVE-2021-44228"], "date_added": [pd.Timestamp("2021-12-10")]}
    )
    with pytest.raises(ValueError, match="published"):
        sync_data.fetch_epss_anchored(kev_only, tmp_path)

