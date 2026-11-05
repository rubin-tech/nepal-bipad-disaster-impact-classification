
"""Shared parsing helpers for the Nepal BIPAD disaster-impact project."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Nepal_Disaster_Data_2020_2026.csv"
)


def dataset_path() -> Path:
    """Return the configured dataset path or the project's default CSV."""
    return Path(
        os.environ.get(
            "BIPAD_CSV_PATH",
            str(DEFAULT_CSV),
        )
    ).expanduser()


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

def load_dataset(path=None) -> pd.DataFrame:
    """Load the merged BIPAD incident/loss dataset."""
    path = Path(path) if path is not None else dataset_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Set BIPAD_CSV_PATH or restore the CSV in data/raw/."
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    return df


# --------------------------------------------------
# 3. Parse geographic coordinates
# --------------------------------------------------

def parse_point(value):
    """
    Extract (longitude, latitude) from a GeoJSON-like point value.

    Supports JSON strings and Python literal dictionary strings.
    Returns (None, None) when coordinates are missing or invalid.
    """
    if not isinstance(value, str) or not value.strip():
        return None, None

    obj = None

    try:
        obj = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        try:
            obj = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return None, None

    try:
        if not isinstance(obj, dict):
            return None, None

        coords = obj.get("coordinates")

        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            return None, None

        longitude = float(coords[0])
        latitude = float(coords[1])

        if not (
            -180 <= longitude <= 180
            and -90 <= latitude <= 90
        ):
            return None, None

        return longitude, latitude

    except (TypeError, ValueError, IndexError, OverflowError):
        return None, None


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Add longitude and latitude columns parsed from the point field."""
    out = df.copy()

    if "point" not in out.columns:
        out["longitude"] = pd.NA
        out["latitude"] = pd.NA
        return out

    coordinates = out["point"].map(parse_point)

    out["longitude"] = coordinates.map(lambda point: point[0])
    out["latitude"] = coordinates.map(lambda point: point[1])

    out["longitude"] = pd.to_numeric(
        out["longitude"],
        errors="coerce",
    )

    out["latitude"] = pd.to_numeric(
        out["latitude"],
        errors="coerce",
    )

    return out


# --------------------------------------------------
# 4. Nepal-local date features
# --------------------------------------------------

def add_local_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Nepal-local date features from incidentOn.

    The original incidentOn column is preserved.
    Naive timestamps are interpreted as UTC, matching the
    existing project's date-handling assumption.
    """
    out = df.copy()

    if "incidentOn" not in out.columns:
        out["incident_datetime_np"] = pd.NaT
        out["incident_year"] = pd.NA
        out["incident_month"] = pd.NA
        out["incident_month_name"] = pd.NA
        out["incident_day_of_year"] = pd.NA
        return out

    dates = pd.to_datetime(
        out["incidentOn"],
        errors="coerce",
        utc=True,
    )

    nepal_dates = dates.dt.tz_convert("Asia/Kathmandu")

    out["incident_datetime_np"] = nepal_dates
    out["incident_year"] = nepal_dates.dt.year
    out["incident_month"] = nepal_dates.dt.month
    out["incident_month_name"] = nepal_dates.dt.month_name()
    out["incident_day_of_year"] = nepal_dates.dt.dayofyear

    return out


# --------------------------------------------------
# 5. Extract hazard and address from incident title
# --------------------------------------------------

def add_title_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hazard_name and incident_address from the incident title.

    Example:
        'Fire at Ram Gaau, Rapti Sonari'
        hazard_name      -> 'Fire'
        incident_address -> 'Ram Gaau, Rapti Sonari'

    If a title does not match the expected pattern, hazard_name
    remains missing. incident_address falls back to streetAddress.
    """
    out = df.copy()

    if "title" not in out.columns:
        out["hazard_name"] = pd.NA
        out["incident_address"] = pd.NA
        return out

    title = (
        out["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    extracted = title.str.extract(
        r"^\s*(?P<hazard_name>.+?)\s+at\s+"
        r"(?P<incident_address>.+?)\s*$",
        flags=re.IGNORECASE,
    )

    out["hazard_name"] = (
        extracted["hazard_name"]
        .str.strip()
        .replace("", pd.NA)
    )

    out["incident_address"] = (
        extracted["incident_address"]
        .str.strip()
        .replace("", pd.NA)
    )

    # Use streetAddress when the title does not provide an address.
    if "streetAddress" in out.columns:
        fallback_address = (
            out["streetAddress"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
        )

        missing_address = (
            out["incident_address"].isna()
            | out["incident_address"].eq("")
        )

        out.loc[missing_address, "incident_address"] = (
            fallback_address.loc[missing_address]
        )

    return out


# --------------------------------------------------
# 6. Project-defined impact classification
# --------------------------------------------------

def impact_class(row) -> str:
    """
    Assign a transparent impact category using recorded outcomes.

    High:
        Any death or missing person;
        >= 5 injured;
        >= 100 affected people;
        or >= 10 destroyed infrastructure units.

    Moderate:
        Any recorded injury, affected people, affected families,
        or destroyed infrastructure when the High criteria do
        not apply.

    Low:
        No recorded outcome meets the criteria above.

    Important:
        If every impact indicator used by the rule is missing, the
        record is left unlabelled (NaN) rather than forced into Low.
        For partially reported records, missing indicators are treated
        as no recorded value for this operational rule. 'Low' therefore
        means low recorded impact among available indicators, not
        confirmed absence of impact. Economic loss is not used.
    """

    impact_columns = [
        "peopleDeathCount", "peopleMissingCount", "peopleInjuredCount",
        "peopleAffectedCount", "familyAffectedCount",
        "infrastructureDestroyedCount",
    ]
    if all(pd.isna(row.get(column)) for column in impact_columns):
        return pd.NA

    def num(column):
        value = pd.to_numeric(
            row.get(column, 0),
            errors="coerce",
        )

        if pd.isna(value):
            return 0.0

        return float(value)

    if (
        num("peopleDeathCount") > 0
        or num("peopleMissingCount") > 0
        or num("peopleInjuredCount") >= 5
        or num("peopleAffectedCount") >= 100
        or num("infrastructureDestroyedCount") >= 10
    ):
        return "High"

    if (
        num("peopleInjuredCount") > 0
        or num("peopleAffectedCount") > 0
        or num("familyAffectedCount") > 0
        or num("infrastructureDestroyedCount") > 0
    ):
        return "Moderate"

    return "Low"
