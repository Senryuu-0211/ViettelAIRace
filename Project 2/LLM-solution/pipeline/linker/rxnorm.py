import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from typing import Optional

from pipeline.config import RXNAV_BASE_URL


def _api_get(url: str, params: dict = None) -> Optional[dict]:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None


def _get_scd_codes(rxcui: str) -> list[str]:
    data = _api_get(
        f"{RXNAV_BASE_URL}/rxcui/{rxcui}/related.json",
        {"tty": "SCD+SBD"},
    )
    if not data:
        return []

    groups = data.get("relatedGroup", {}).get("conceptGroup", [])
    codes = []
    for g in groups:
        for p in g.get("conceptProperties", []):
            code = p.get("rxcui", "")
            if code:
                codes.append(code)
    return codes


def _search_rxnorm(query: str) -> Optional[str]:
    """Search RxNorm and return best matching rxcui, or None."""
    data = _api_get(
        f"{RXNAV_BASE_URL}/approximateTerm.json",
        {"term": query, "maxEntries": 3},
    )
    if not data:
        return None

    candidates = data.get("approximateGroup", {}).get("candidate", [])
    best = None
    best_score = float("-inf")
    for c in candidates:
        rxcui = c.get("rxcui")
        if not rxcui:
            continue
        score = float(c.get("score", 0))
        if score > best_score:
            best_score = score
            best = rxcui
    return best


def _parse_strength(drug_name: str) -> Optional[str]:
    """Extract numeric strength from drug name."""
    m = re.search(r"(\d+\.?\d*\s*(?:MG|G|MCG|ML|%))", drug_name, re.IGNORECASE)
    return m.group(1) if m else None


def _active_ingredient(drug_name: str) -> str:
    """Get first word(s) as active ingredient."""
    parts = drug_name.split()
    if not parts:
        return drug_name
    return parts[0]


def lookup_drug(drug_name: str, drug: dict = None) -> list[str]:
    if drug:
        ingredient = (drug.get("ingredient") or "").strip()
        strength = (drug.get("strength") or "").strip()
        if ingredient:
            search = f"{ingredient} {strength}".strip()
            rxcui = _search_rxnorm(search)
            if not rxcui:
                rxcui = _search_rxnorm(ingredient)
            if rxcui:
                scd_codes = _get_scd_codes(rxcui)
                return scd_codes if scd_codes else [rxcui]

    rxcui = _search_rxnorm(drug_name)

    if not rxcui:
        ingredient = _active_ingredient(drug_name)
        rxcui = _search_rxnorm(ingredient)

    if not rxcui:
        return []

    scd_codes = _get_scd_codes(rxcui)
    if scd_codes:
        return scd_codes
    return [rxcui]


def lookup_batch(drug_names: list[str]) -> dict[str, list[str]]:
    result = {}
    for name in drug_names:
        codes = lookup_drug(name)
        if codes:
            result[name] = codes
        time.sleep(0.3)
    return result
