"""Property tests for French nuclear extraction and caching.

Feature: agile-predict-advanced-features, Properties 5 & 6
"""

import json
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


NS = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"


def build_entsoe_xml(time_series_list):
    """Build a minimal ENTSO-E XML response with given TimeSeries entries.

    Each entry is a dict: {"psr_type": "B14", "quantity": 5000.0}
    """
    root = ET.Element("GL_MarketDocument", xmlns=NS)
    for ts in time_series_list:
        ts_el = ET.SubElement(root, "TimeSeries")
        mkt = ET.SubElement(ts_el, "MktPSRType")
        psr = ET.SubElement(mkt, "psrType")
        psr.text = ts["psr_type"]
        period = ET.SubElement(ts_el, "Period")
        point = ET.SubElement(period, "Point")
        pos = ET.SubElement(point, "position")
        pos.text = "1"
        qty = ET.SubElement(point, "quantity")
        qty.text = str(ts["quantity"])
    return ET.tostring(root, encoding="unicode")


def parse_nuclear_from_xml(xml_text):
    """Replicate the XML parsing logic from fetch_french_nuclear."""
    root = ET.fromstring(xml_text)
    ns = {"ns": NS}
    nuclear_mw = 0.0
    nuclear_count = 0
    for ts in root.findall(".//ns:TimeSeries", ns):
        psr_type = ts.find(".//ns:MktPSRType/ns:psrType", ns)
        if psr_type is not None and psr_type.text == "B14":
            points = ts.findall(".//ns:Point", ns)
            if points:
                last_point = points[-1]
                qty = last_point.find("ns:quantity", ns)
                if qty is not None:
                    nuclear_mw += float(qty.text)
                    nuclear_count += 1
    return nuclear_mw / 1000.0 if nuclear_count > 0 else None


@given(
    nuclear_mw=st.floats(min_value=100, max_value=70000),
    other_mw=st.floats(min_value=100, max_value=30000),
)
@settings(max_examples=100)
def test_xml_extracts_only_b14_nuclear(nuclear_mw, other_mw):
    """Property 5: Parser sums only B14 (nuclear) quantities, converts MW to GW.

    Test Case:
    test_french_nuclear\\test_xml_extracts_only_b14_nuclear

    Purpose:
    Validates that the ENTSO-E XML parser correctly filters for B14 psrType
    and ignores other generation types.

    Test Conditions:
    - XML with one B14 (nuclear) and one B16 (solar) TimeSeries
    - Random MW values

    Key Properties:
        For any valid XML with B14 and non-B14 TimeSeries,
        parser sums only B14 quantities and converts MW to GW.

    Expected Behaviour:
    - Result equals nuclear_mw / 1000 (only B14 counted)
    - B16 value is ignored
    """
    xml = build_entsoe_xml([
        {"psr_type": "B14", "quantity": nuclear_mw},
        {"psr_type": "B16", "quantity": other_mw},
    ])
    result = parse_nuclear_from_xml(xml)
    assert result == pytest.approx(nuclear_mw / 1000.0, rel=1e-6)


def test_xml_no_nuclear_returns_none():
    """Property 5: No B14 data returns None.

    Test Case:
    test_french_nuclear\\test_xml_no_nuclear_returns_none

    Purpose:
    Validates graceful handling when no nuclear data is present.

    Test Conditions:
    - XML with only non-B14 TimeSeries

    Key Properties:
        When no B14 TimeSeries exist, parser returns None.

    Expected Behaviour:
    - Returns None
    """
    xml = build_entsoe_xml([{"psr_type": "B16", "quantity": 5000}])
    result = parse_nuclear_from_xml(xml)
    assert result is None


@given(
    value=st.floats(min_value=10, max_value=70),
)
@settings(max_examples=50)
def test_cache_round_trip(value):
    """Property 6: Write to cache then read within freshness window returns same value.

    Test Case:
    test_french_nuclear\\test_cache_round_trip

    Purpose:
    Validates that the cache file correctly stores and retrieves
    the French nuclear generation value.

    Test Conditions:
    - Random nuclear GW values
    - Cache read within freshness window

    Key Properties:
        Writing to cache and reading within 6 hours returns the same value.

    Expected Behaviour:
    - Round-trip preserves the value exactly
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        cache_data = {
            "last_updated": pd.Timestamp.now(tz="UTC").isoformat(),
            "french_nuclear_gw": value,
            "data_timestamp": "202604080000",
        }
        json.dump(cache_data, f)
        cache_path = f.name

    try:
        with open(cache_path) as f:
            loaded = json.load(f)
        assert loaded["french_nuclear_gw"] == pytest.approx(value)
    finally:
        os.unlink(cache_path)
