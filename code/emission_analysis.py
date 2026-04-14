"""
emission_analysis.py
────────────────────
Quantitative transport emission analysis for the
Hybrid ML + LLM Decision Support System for Sustainable Sri Lankan Tourism.

Computes per-passenger emissions, counterfactual mode comparisons, and
reduction estimates — all injected into the Ollama LLM prompt to produce
numeric, evidence-based, transport-specific recommendations.

No ML model is used here; these are derived entirely from the trip input
features already collected by the Streamlit UI.
"""

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# Sri Lanka emission factors  (kgCO₂ per PASSENGER-km)
#
# All values represent the standardised unit g CO₂ / pax-km (÷ 1000 = kgCO₂).
# This allows direct cross-mode comparison regardless of vehicle capacity.
#
# Key Citations
# ─────────────
# Private Car (0.171 kgCO₂/pax-km = 171 g/pax-km, single occupant):
#   UK DESNZ (2023) "Greenhouse Gas Reporting: Conversion Factors", Table 1.
#   Medium petrol car, 1 occupant (driver only) — most conservative typical
#   case.  IPCC AR6 WG3 (2022) Table 10.SM.7 confirms 100–250 g/pax-km range
#   for road passenger transport globally.
# Train SLR   (0.035 kgCO₂/pax-km = 35 g/pax-km, diesel fleet):
#   Sri Lanka Railways operational data; average passenger loading on
#   inter-city services.  The IEA 2023 South Asian rail average (14 g/pax-km)
#   is used separately as the cross-mode comparison baseline — see
#   TRAIN_BASELINE_KG_PER_PKM below.
# ─────────────────────────────────────────────────────────────────────────────
EMISSION_FACTORS_PKM: dict[str, float] = {
    "Private Car":               0.171,   # 171 g/pax-km — DESNZ 2023, 1-occupant petrol car
    "Tourist Van / Minibus":     0.052,   # diesel, ~12 pax typical load
    "Tuk-tuk (Three-wheeler)":   0.092,   # 2/4-stroke, avg 1.5 pax
    "Electric Tuk-tuk (e-tuk)":  0.018,   # Sri Lanka grid 2024 (0.59 kgCO₂/kWh)
    "Public Bus (CTB)":          0.039,   # inter-city, avg 40 pax load
    "Train":                     0.035,   # 35 g/pax-km — Sri Lanka Railways diesel fleet
    "Motorcycle":                0.110,   # 125–150 cc, single rider
    "Bicycle":                   0.000,   # zero operational emissions
}

# Fixed comparison set — always evaluated regardless of the current vehicle
_COMPARISON_MODES = ["Train", "Public Bus (CTB)", "Electric Tuk-tuk (e-tuk)"]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CounterfactualOption:
    """One alternative transport mode and its emission comparison."""
    name: str
    per_passenger_kgCO2: float    # absolute kgCO₂ for this mode on the same route
    reduction_pct: float           # positive = saving vs current; negative = worse
    is_better: bool                # True if greener than current vehicle
    ef_per_pkm: float              # emission factor used (kgCO₂/pax-km)


@dataclass
class TransportMetrics:
    """Complete quantitative transport profile for one trip."""

    # ── raw inputs ────────────────────────────────────────────────────────
    vehicle_type: str
    transport_total_kgCO2: float
    distance_km: float
    occupancy: int

    # ── per-passenger ─────────────────────────────────────────────────────
    per_passenger_kgCO2: float
    kgCO2_per_km: float
    per_passenger_per_km: float

    # ── counterfactual alternatives ───────────────────────────────────────
    alternatives: list            # list[CounterfactualOption]

    # ── occupancy scaling ─────────────────────────────────────────────────
    double_occ_per_passenger_kgCO2: float
    reduction_pct_double_occupancy: float

    # ── summary ───────────────────────────────────────────────────────────
    best_alternative: object      # CounterfactualOption | None — highest % saving
    already_optimal: bool         # True when current vehicle is already the greenest


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reduction_pct(current: float, alternative: float) -> float:
    """
    Return the percentage reduction from ``current`` to ``alternative``.

    Formula:  reduction% = (current - alternative) / current × 100

    Return value contract
    ---------------------
    +100.0  → alternative is completely zero-emission (e.g. Bicycle)
        0.0 → current is zero, or values are equal
      < 0.0 → alternative is *worse* than current (no saving possible)
    Clamped to [-∞, 100.0] — cannot exceed 100% saving.
    Division-by-zero guarded: returns 0.0 when current <= 0.
    """
    if current <= 0:
        return 0.0
    raw = (current - alternative) / current * 100
    return round(min(raw, 100.0), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_transport_metrics(trip_data: dict, vehicle_type: str) -> TransportMetrics:
    """
    Derive all quantitative transport metrics from the ML model's input features.

    Parameters
    ----------
    trip_data    : dict
        The same ``input_dict`` passed to the scikit-learn model in app.py.
    vehicle_type : str
        The vehicle selected in the Streamlit selectbox.

    Returns
    -------
    TransportMetrics
        Per-passenger emissions, counterfactual comparisons, and occupancy
        scaling values — ready to be embedded in the Ollama LLM prompt.
    """
    # All three denominators are guarded against zero:
    #   distance_km → clamped to ≥ 0.1 km
    #   occupancy   → clamped to ≥ 1 person
    # transport_total is clamped to ≥ 0.0 so negative UI input cannot
    # produce negative per-passenger or per-km values.
    transport_total = max(float(trip_data.get("transport_emissions_kgCO2", 0.0)), 0.0)  # kgCO₂
    distance_km     = max(float(trip_data.get("distance_km", 1.0)), 0.1)               # km
    occupancy       = max(int(trip_data.get("occupancy", 1)), 1)                       # persons

    per_passenger        = transport_total / occupancy                   # kgCO₂ / pax
    kgCO2_per_km         = transport_total / distance_km                 # kgCO₂ / km (vehicle total)
    per_passenger_per_km = per_passenger / distance_km                   # kgCO₂ / pax-km

    # ── Counterfactual alternatives ───────────────────────────────────────
    alternatives: list[CounterfactualOption] = []
    for mode in _COMPARISON_MODES:
        if mode == vehicle_type:
            continue                          # skip self-comparison
        ef     = EMISSION_FACTORS_PKM[mode]
        alt_pp = round(ef * distance_km, 5)
        red    = _reduction_pct(per_passenger, alt_pp)
        alternatives.append(CounterfactualOption(
            name=mode,
            per_passenger_kgCO2=alt_pp,
            reduction_pct=red,
            is_better=red > 0,
            ef_per_pkm=ef,
        ))

    # ── Occupancy doubling ────────────────────────────────────────────────
    double_pp  = round(transport_total / (occupancy * 2), 5)
    red_double = _reduction_pct(per_passenger, double_pp)

    # ── Best greener alternative (highest % saving) ───────────────────────
    better = [a for a in alternatives if a.is_better]
    best   = max(better, key=lambda a: a.reduction_pct) if better else None

    return TransportMetrics(
        vehicle_type=vehicle_type,
        transport_total_kgCO2=round(transport_total, 5),
        distance_km=round(distance_km, 2),
        occupancy=occupancy,
        per_passenger_kgCO2=round(per_passenger, 5),
        kgCO2_per_km=round(kgCO2_per_km, 5),
        per_passenger_per_km=round(per_passenger_per_km, 7),
        alternatives=alternatives,
        double_occ_per_passenger_kgCO2=double_pp,
        reduction_pct_double_occupancy=red_double,
        best_alternative=best,
        already_optimal=best is None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level-2 DSS — Rule-Based Layer
# ─────────────────────────────────────────────────────────────────────────────

# ── Train / public-transport baseline constant ───────────────────────────────
#
# Source : IEA (2023) "CO₂ Emissions from Fuel Combustion", p. 185.
#          Table: CO₂ intensity of passenger transport by mode — South Asia.
# Cross-validated by:
#   • Our World in Data, Hannah Ritchie (2020) —
#     "Which form of transport has the smallest carbon footprint?"
#     https://ourworldindata.org/travel-carbon-footprint
#   • IPCC AR6 WG3 (2022), Table 10.SM.7 — rail vs road comparison.
#   • Sri Lanka Railways operational data — validates South Asian rail range.
# Value  : 14 g CO₂ per passenger-kilometre  (= 0.014 kg CO₂ / pax-km)
#
# Formula used in Metric 3
# ────────────────────────
#   E_train  =  EF_train  ×  d
#
#   where:
#     E_train   = train emission for this trip  [kgCO₂ per passenger]
#     EF_train  = 0.014  kgCO₂ / pax-km        [IEA 2023 constant]
#     d         = route distance in km          [from user input]
#
# Unit conversion:
#   TRAIN_BASELINE_G_PER_PKM  = 14.0  g / pax-km   (human-readable form)
#   TRAIN_BASELINE_KG_PER_PKM = 14.0 ÷ 1000         (used in arithmetic)
#
# Interpretation:
#   If the tourist's current per-passenger rate exceeds 14 g/pax-km,
#   switching to rail would reduce their transport emission.  The saving is
#   reported as both an absolute kgCO₂ difference and a percentage.
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_BASELINE_G_PER_PKM: float  = 14.0
TRAIN_BASELINE_KG_PER_PKM: float = TRAIN_BASELINE_G_PER_PKM / 1000   # 0.014 kgCO₂/pax-km

# “Full vehicle” occupancy scenario used in Metric 2
FULL_OCCUPANCY_SCENARIO: int = 4


@dataclass
class RuleBasedMetrics:
    """
    The four explicit DSS metrics computed entirely in Python before any LLM
    call is made. Every field is a plain, traceable number — no ML black-box.

    Data flow:
        User inputs → run_rule_based_analysis() → RuleBasedMetrics
        RuleBasedMetrics → build_dss_prompt() → Ollama prompt string
    """

    # ── Inputs (echoed for traceability) ───────────────────────────────────
    emission_level:         str     # ML label: 'low' | 'medium' | 'high'
    vehicle_type:           str
    distance_km:            float
    current_passengers:     int
    total_transport_kgCO2:  float

    # ── Metric 1: Per-passenger emission ─────────────────────────────────
    per_passenger_kgCO2:    float
    per_passenger_g_per_km: float   # human-readable grams / pax-km

    # ── Metric 2: Full-occupancy scenario (FULL_OCCUPANCY_SCENARIO = 4 pax) ──
    full_occ_passengers:          int
    full_occ_per_passenger_kgCO2: float
    full_occ_reduction_pct:       float
    full_occ_g_per_km:            float   # standardised rate: g CO₂/pax-km at full occupancy

    # ── Metric 3: Train / public-transport baseline (14 g/pax-km constant) ──
    train_baseline_g_per_pkm:       float   # = 14.0
    train_baseline_kgCO2:           float   # 0.014 × distance_km
    vs_train_kgCO2_diff:            float   # + = current is worse than train
    vs_train_saving_pct:            float   # + = % saved by switching to train
    current_is_greener_than_train:  bool

    # ── Metric 4: Efficiency label ─────────────────────────────────────
    efficiency_label: str   # 'EFFICIENT' | 'MODERATE' | 'INEFFICIENT'
    efficiency_note:  str


def run_rule_based_analysis(
    trip_data:      dict,
    vehicle_type:   str,
    emission_level: str,
) -> RuleBasedMetrics:
    """
    Rule-Based Layer: derive the four DSS metrics from raw trip input data.

    Runs entirely in Python — no ML model, no LLM call. The results are
    injected verbatim into the Ollama prompt so every LLM recommendation
    is grounded in verifiable, pre-computed numbers.

    Parameters
    ----------
    trip_data      : dict  — same dict passed to the scikit-learn model.
    vehicle_type   : str   — Streamlit selectbox value.
    emission_level : str   — ML model prediction ('low' | 'medium' | 'high').
    """
    # All denominators guarded against zero (same pattern as compute_transport_metrics)
    total       = max(float(trip_data.get("transport_emissions_kgCO2", 0.0)), 0.0)  # kgCO₂
    distance_km = max(float(trip_data.get("distance_km", 1.0)), 0.1)               # km
    passengers  = max(int(trip_data.get("occupancy", 1)), 1)                       # persons

    # ── Metric 1: Per-passenger ────────────────────────────────────────────
    # Formula: E_pax = E_total / N
    #   E_pax     [kgCO₂/pax]    = total vehicle emission ÷ number of passengers
    #   g_per_km  [g CO₂/pax-km] = E_pax ÷ distance × 1000
    #                              (÷ distance: normalise to per-km)
    #                              (× 1000: convert kg → g for human-readable display)
    per_pax     = total / passengers                       # kgCO₂ / pax
    per_pax_gpk = (per_pax / distance_km) * 1000          # g CO₂ / pax-km  (kg÷km × 1000 = g/km)

    # ── Metric 2: Full-occupancy scenario ─────────────────────────────────
    # Formula: E_full = E_total / N_full
    #   N_full is the benchmark seat count (4); clamped so it never drops
    #   below the actual passenger count (avoids a negative "saving").
    full_occ     = max(FULL_OCCUPANCY_SCENARIO, passengers)  # persons; always ≥ current
    full_occ_pp  = total / full_occ                          # kgCO₂ / pax at full capacity
    full_occ_red = _reduction_pct(per_pax, full_occ_pp)     # % saving vs current occupancy
    # Standardised rate (g/pax-km) so full-occ can be directly compared to
    # the train 14 g/pax-km baseline and the current g/pax-km rate.
    full_occ_gpk = round((full_occ_pp / distance_km) * 1000, 2)  # g CO₂/pax-km at full occ

    # ── Metric 3: Train baseline ───────────────────────────────────────────
    # Formula: E_train = EF_train × d  (see constant block above for full derivation)
    train_kgco2   = round(TRAIN_BASELINE_KG_PER_PKM * distance_km, 5)  # kgCO₂ / pax
    vs_train_diff = round(per_pax - train_kgco2, 5)   # kgCO₂/pax; + means current is worse
    vs_train_pct  = _reduction_pct(per_pax, train_kgco2)               # % saving if switched
    is_greener    = per_pax <= train_kgco2                              # True → already better

    # ── Metric 4: Efficiency label ────────────────────────────────────────
    if per_pax_gpk <= 20:
        eff_label = "EFFICIENT"
        eff_note  = (
            f"{per_pax_gpk:.1f} g/pax-km — on par with or below "
            "the 14 g/pax-km train baseline"
        )
    elif per_pax_gpk <= 60:
        eff_label = "MODERATE"
        eff_note  = (
            f"{per_pax_gpk:.1f} g/pax-km — above the 14 g/pax-km "
            "train baseline; improvement is achievable"
        )
    else:
        eff_label = "INEFFICIENT"
        eff_note  = (
            f"{per_pax_gpk:.1f} g/pax-km — well above the "
            "14 g/pax-km train baseline"
        )

    return RuleBasedMetrics(
        emission_level=emission_level,
        vehicle_type=vehicle_type,
        distance_km=round(distance_km, 2),
        current_passengers=passengers,
        total_transport_kgCO2=round(total, 5),
        per_passenger_kgCO2=round(per_pax, 5),
        per_passenger_g_per_km=round(per_pax_gpk, 2),
        full_occ_passengers=full_occ,
        full_occ_per_passenger_kgCO2=round(full_occ_pp, 5),
        full_occ_reduction_pct=full_occ_red,
        full_occ_g_per_km=full_occ_gpk,
        train_baseline_g_per_pkm=TRAIN_BASELINE_G_PER_PKM,
        train_baseline_kgCO2=train_kgco2,
        vs_train_kgCO2_diff=vs_train_diff,
        vs_train_saving_pct=vs_train_pct,
        current_is_greener_than_train=is_greener,
        efficiency_label=eff_label,
        efficiency_note=eff_note,
    )


def build_dss_prompt(rbm: RuleBasedMetrics, location: str = "Sri Lanka") -> str:
    """
    Construct the Ollama prompt from RuleBasedMetrics.

    Key design change (v3 — small-model safe)
    ------------------------------------------
    NUMBERS lines are pre-written entirely in Python using the verified
    RuleBasedMetrics values.  The LLM is told to copy them unchanged and
    is only asked to supply:
      - ACTION  : a short descriptive title  (text, no numbers)
      - TIP     : 1-2 sentences naming a real Sri Lankan service

    This eliminates hallucinated numbers entirely: the LLM cannot invent
    a saving percentage because there is no blank for it to fill.
    sanitize_numbers_lines() in ollama_recommendations.py acts as a
    second safety net.
    """
    # ── Pre-compute the two NUMBERS strings ──────────────────────────────────
    # These are injected verbatim — the LLM must copy them unchanged.
    # Format: current_rate → alternative_rate → save X%
    _n_train = (
        f"{rbm.per_passenger_g_per_km:.1f} g/pax-km "
        f"→ {rbm.train_baseline_g_per_pkm:.1f} g/pax-km "
        f"→ save {rbm.vs_train_saving_pct}%"
    )
    _n_occ = (
        f"{rbm.per_passenger_g_per_km:.1f} g/pax-km "
        f"→ {rbm.full_occ_g_per_km:.1f} g/pax-km "
        f"→ save {rbm.full_occ_reduction_pct}%"
    )

    # Assign numbers + action hints per slot
    if rbm.current_is_greener_than_train:
        n1, hint1 = _n_occ,   "share vehicle / fill all seats"
        n2, hint2 = _n_occ,   "use shared tuk-tuk or group transport"
        n3, hint3 = _n_occ,   "carpool or join organised tour group"
    else:
        n1, hint1 = _n_train, "switch to train or public transport"
        n2, hint2 = _n_occ,   "share vehicle / fill to full occupancy"
        n3, hint3 = _n_train, "use CTB bus or e-tuk instead of car"

    prompt = f"""You are a sustainable transport advisor for tourists in {location}, Sri Lanka.
Fill in the [ACTION] and [TIP] fields below. The NUMBERS lines are already written
— copy them EXACTLY into your answer, do not change any digit or symbol.

Trip: {rbm.vehicle_type} | {rbm.distance_km} km | {rbm.current_passengers} passenger(s)
Current rate: {rbm.per_passenger_g_per_km:.1f} g/pax-km | ML level: {rbm.emission_level.upper()} | Rating: {rbm.efficiency_label}

--- OUTPUT TEMPLATE (fill ACTION and TIP only) ---

1. ACTION: [{hint1} — write a 4-6 word title, no numbers]
   NUMBERS: {n1}
   TIP: [1-2 sentences naming a real service or route in {location}]

2. ACTION: [{hint2} — write a 4-6 word title, no numbers]
   NUMBERS: {n2}
   TIP: [1-2 sentences naming a real service or route in {location}]

3. ACTION: [{hint3} — write a 4-6 word title, no numbers]
   NUMBERS: {n3}
   TIP: [1-2 sentences naming a real service or route in {location}]

VERDICT: [One sentence: most impactful action and its saving % from above]

Rules: Transport only. Name real Sri Lankan options (Ella Odyssey train, PickMe e-tuk,
CTB bus, shared jeep at Yala). Do not repeat the same action twice.
"""
    return prompt
