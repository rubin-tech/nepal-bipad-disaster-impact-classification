import pandas as pd
from src.data_utils import impact_class

FIELDS = [
    "peopleDeathCount", "peopleMissingCount", "peopleInjuredCount",
    "peopleAffectedCount", "familyAffectedCount", "infrastructureDestroyedCount",
]

def row(**kwargs):
    values = {field: 0 for field in FIELDS}
    values.update(kwargs)
    return pd.Series(values)

def test_all_impact_fields_missing_is_unlabelled():
    assert pd.isna(impact_class(pd.Series({field: pd.NA for field in FIELDS})))

def test_high_thresholds():
    assert impact_class(row(peopleDeathCount=1)) == "High"
    assert impact_class(row(peopleMissingCount=1)) == "High"
    assert impact_class(row(peopleInjuredCount=5)) == "High"
    assert impact_class(row(peopleAffectedCount=100)) == "High"
    assert impact_class(row(infrastructureDestroyedCount=10)) == "High"

def test_moderate_thresholds():
    assert impact_class(row(peopleInjuredCount=1)) == "Moderate"
    assert impact_class(row(peopleAffectedCount=1)) == "Moderate"
    assert impact_class(row(familyAffectedCount=1)) == "Moderate"
    assert impact_class(row(infrastructureDestroyedCount=1)) == "Moderate"

def test_low_when_all_reported_indicators_are_zero():
    assert impact_class(row()) == "Low"
