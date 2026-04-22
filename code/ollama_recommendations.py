"""
ollama_recommendations.py
--------------------------
Generates smart, localised Sri Lankan carbon-footprint recommendations
using Ollama (gemma3:1b) based on:
  - The ML model's emission classification (low / medium / high)
  - The tourist's current location in Sri Lanka
  - The vehicle type used

Falls back to static, Sri Lanka-specific recommendations if Ollama is
unavailable.
"""

import re
from typing import Optional, List

import ollama
from emission_analysis import TransportMetrics, EMISSION_FACTORS_PKM, RuleBasedMetrics, build_dss_prompt


# ---------------------------------------------------------------------------
# Location-specific context for the prompt
# ---------------------------------------------------------------------------

# Keyed by the exact strings used in the Streamlit selectbox.
_LOCATION_CONTEXT: dict[str, str] = {
    "Colombo": (
        "Colombo has a growing network of electric three-wheelers (e-tuks) bookable via "
        "PickMe and inDriver. The city is congested, so carpooling via PickMe Car or the "
        "urban rail line to Maradana/Fort reduces per-person emissions significantly."
    ),
    "Kandy": (
        "Kandy is well connected by train from Colombo (a scenic, low-emission route). "
        "Within the city, shared three-wheelers and the local CTB bus network are the "
        "most sustainable options. Avoid private vans for short hops around Kandy Lake."
    ),
    "Ella": (
        "The Ella Odyssey (Kandy–Ella or Colombo–Ella train) is one of the world's most "
        "scenic rail journeys and has a carbon footprint ~80% lower than a private car "
        "over the same route. Ella has several vegan-friendly 'rice and curry' spots that "
        "serve local, low-emission produce."
    ),
    "Galle": (
        "Galle is reachable by the Southern Expressway bus (low-emission coach) or the "
        "Coastal Line train from Colombo Fort. Inside the Galle Fort area, walking and "
        "cycling are the only sustainable options — private tuk-tuks in the Fort add "
        "unnecessary emissions over very short distances."
    ),
    "Nuwara Eliya": (
        "Nuwara Eliya sits at high altitude; the terrain factor is high, making private "
        "vehicles burn significantly more fuel. The hill-country train from Kandy to "
        "Nanu Oya (nearest station) is the greenest choice. Many estate bungalows run "
        "on hydropower — ask your host about the energy source."
    ),
    "Sigiriya": (
        "Sigiriya is best reached by shared minibus or the Dambulla CTB bus from Colombo. "
        "Once there, cycling is available for hire and covers most nearby sites (Pidurangala "
        "Rock, Minneriya) with zero emissions. Avoid air-conditioned private vans for "
        "day trips — they are the single biggest emitter in this region."
    ),
    "Trincomalee": (
        "Trincomalee is served by a direct train from Colombo (Batticaloa line) — an "
        "easy, low-emission overnight option. Beach areas can be explored by bicycle "
        "or pedal boat. Local seafood 'kottu roti' shops use minimal packaging compared "
        "to tourist restaurants."
    ),
    "Yala": (
        "Yala is remote; most visitors arrive by private vehicle. Joining a shared jeep "
        "safari (split among 6 passengers) cuts per-person transport emissions by up to "
        "80% vs a solo hire. Choose eco-lodges with solar panels to offset the "
        "generator-heavy accommodation typical in this area."
    ),
    "Arugam Bay": (
        "Arugam Bay has a strong surf-backpacker culture; many guesthouses are moving "
        "to solar. Rent a bicycle or a pedal-assisted e-bike to reach Pottuvil Lagoon "
        "and nearby surf points instead of tuk-tuks. Reduce food emissions by choosing "
        "local 'rice and curry' over imported Western dishes."
    ),
    "Anuradhapura": (
        "Anuradhapura's sacred sites are spread over a large area; bicycle hire is the "
        "traditional and most eco-friendly way to tour them. Electric golf-cart tours "
        "are also available near the main dagobas. Avoid generator-reliant guesthouses "
        "outside the main town — solar-powered lodges are available near Nuwarawewa."
    ),
}

_DEFAULT_LOCATION_CONTEXT = (
    "Sri Lanka has an expanding network of low-emission trains, CTB buses, and "
    "electric three-wheelers. Wherever you are, sharing transport and choosing "
    "local produce are the two most impactful changes you can make."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_quantitative_prompt(
    emission_level: str,
    trip_data: dict,
    location: str,
    vehicle_type: str,
    metrics: TransportMetrics,
) -> str:
    """Build a quantitative, transport-focused prompt that embeds pre-computed metrics."""
    location_ctx = _LOCATION_CONTEXT.get(location, _DEFAULT_LOCATION_CONTEXT)

    # ── Alternatives table ────────────────────────────────────────────────
    alt_lines = []
    for alt in metrics.alternatives:
        tag = f"SAVE {alt.reduction_pct}%" if alt.is_better else f"WORSE (+{abs(alt.reduction_pct)}%)"
        alt_lines.append(
            f"  {alt.name:<32} {alt.per_passenger_kgCO2:.5f} kgCO₂/pax "
            f"[ef={alt.ef_per_pkm} kgCO₂/pax-km]  →  {tag}"
        )
    alts_block = "\n".join(alt_lines) if alt_lines else "  (no standard alternatives differ from current vehicle)"

    double_tag = (
        f"SAVE {metrics.reduction_pct_double_occupancy}%"
        if metrics.reduction_pct_double_occupancy > 0
        else "already optimal"
    )

    best_summary = (
        f"  Best greener option: {metrics.best_alternative.name} → "
        f"{metrics.best_alternative.per_passenger_kgCO2:.5f} kgCO₂/pax → "
        f"SAVE {metrics.best_alternative.reduction_pct}%"
        if metrics.best_alternative
        else "  Current vehicle is already the lowest-emission option for this route."
    )

    prompt = f"""You are a Sri Lankan Sustainable Tourism Expert and a data-driven Decision Support System.
You ONLY advise on TRANSPORT emissions. Do NOT mention food, hotels, or plastic.

══════════════════════════════════════════════════
  QUANTITATIVE TRANSPORT ANALYSIS
  Generated by ML model + Python emission engine
══════════════════════════════════════════════════

Trip Profile:
  Location   : {location}
  Vehicle    : {vehicle_type}
  Distance   : {metrics.distance_km} km
  Passengers : {metrics.occupancy}
  Trip days  : {trip_data.get('trip_days', 'N/A')}
  ML Class   : {emission_level.upper()} carbon footprint

Current Transport Emissions (from ML model input):
  Total vehicle emission   : {metrics.transport_total_kgCO2:.5f} kgCO₂
  Per passenger            : {metrics.per_passenger_kgCO2:.5f} kgCO₂
  Per passenger per km     : {metrics.per_passenger_per_km:.7f} kgCO₂/km

Counterfactual Alternatives (same {metrics.distance_km} km route):
{alts_block}
  Double occupancy ({metrics.occupancy}→{metrics.occupancy * 2} pax, same vehicle)
      {metrics.double_occ_per_passenger_kgCO2:.5f} kgCO₂/pax  →  {double_tag}

{best_summary}

Local transport context for {location}:
{location_ctx}

══════════════════════════════════════════════════
  YOUR OUTPUT — follow this format EXACTLY
══════════════════════════════════════════════════
Write exactly 3 numbered transport recommendations using ONLY the numbers from the analysis above.

1. ACTION: [concise action name]
   NUMBERS: {metrics.per_passenger_kgCO2:.5f} kgCO₂/pax → [value from table above] kgCO₂/pax → save [X]%
   TIP: [1-2 sentences naming a specific Sri Lankan service or route in {location}]

2. ACTION: [concise action name]
   NUMBERS: {metrics.per_passenger_kgCO2:.5f} kgCO₂/pax → [value from table above] kgCO₂/pax → save [X]%
   TIP: [1-2 sentences naming a specific Sri Lankan service or route in {location}]

3. ACTION: [concise action name]
   NUMBERS: {metrics.per_passenger_kgCO2:.5f} kgCO₂/pax → [value from table above] kgCO₂/pax → save [X]%
   TIP: [1-2 sentences naming a specific Sri Lankan service or route in {location}]

VERDICT: [One sentence: the single highest-impact transport action with its exact % saving]

Rules:
- Use ONLY the numbers already computed above. Do NOT invent or recalculate.
- Transport only. No food, accommodation, or waste.
- If current vehicle is already optimal, all 3 tips must focus on increasing occupancy.
- Name specific Sri Lankan options (Ella Odyssey train, PickMe, CTB bus, e-tuk, shared jeep).
- Do NOT repeat the same action twice.
"""
    return prompt


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def parse_recommendation_output(text: str) -> Optional[dict]:
    """
    Parse the structured LLM output into a Python dict.

    Expected input format (from the quantitative prompt):
        1. ACTION: ...
           NUMBERS: X kgCO₂/pax → Y kgCO₂/pax → save Z%
           TIP: ...
        2. ...
        3. ...
        VERDICT: ...

    Returns
    -------
    {
        "recommendations": [
            {"action": str, "numbers": str, "tip": str},
            ...
        ],
        "verdict": str
    }
    or ``None`` if parsing fails (caller falls back to raw markdown display).
    """
    try:
        recs: List[dict] = []

        # Split on lines that start a new numbered item (1. / 2. / 3.)
        blocks = re.split(r"\n(?=\d+\.\s)", text.strip())

        for block in blocks:
            if not re.match(r"^\d+\.\s", block.strip()):
                continue
            action_m  = re.search(r"ACTION:\s*(.+?)(?:\n|$)",              block, re.IGNORECASE)
            numbers_m = re.search(r"NUMBERS:\s*(.+?)(?:\n|$)",             block, re.IGNORECASE)
            tip_m     = re.search(r"TIP:\s*([\s\S]+?)(?=\n\s*\d+\.\s|\nVERDICT:|$)", block, re.IGNORECASE)

            if action_m:
                recs.append({
                    "action":  action_m.group(1).strip(),
                    "numbers": numbers_m.group(1).strip() if numbers_m else "",
                    "tip":     tip_m.group(1).strip()     if tip_m     else "",
                })

        verdict_m = re.search(r"VERDICT:\s*([\s\S]+?)$", text, re.IGNORECASE)
        verdict   = verdict_m.group(1).strip() if verdict_m else ""

        return {"recommendations": recs, "verdict": verdict} if recs else None

    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# DSS Integrity: hallucination detection
# ---------------------------------------------------------------------------

def _build_allowed_numbers(rbm: RuleBasedMetrics) -> set[float]:
    """
    Build the complete set of numeric values injected into the Ollama prompt.

    Includes rounded variants at 0–5 decimal places to handle any LLM
    formatting artefacts (e.g. the LLM writing 0.037 instead of 0.03700).
    """
    raw: list[float] = [
        rbm.per_passenger_kgCO2,
        rbm.full_occ_per_passenger_kgCO2,
        float(rbm.full_occ_reduction_pct),
        rbm.train_baseline_kgCO2,
        float(rbm.vs_train_saving_pct),
        rbm.total_transport_kgCO2,
        abs(rbm.vs_train_kgCO2_diff),
        float(rbm.distance_km),
        float(rbm.current_passengers),
        float(rbm.full_occ_passengers),
        float(rbm.per_passenger_g_per_km),
        float(rbm.full_occ_g_per_km),          # REF-H — added so detector won't false-flag
        float(rbm.train_baseline_g_per_pkm),   # REF-I — 14.0 g/pax-km
        14.0,    # IEA train baseline constant (g/pax-km)
        0.014,   # IEA train baseline constant (kg/pax-km)
    ]
    allowed: set[float] = set()
    for v in raw:
        for decimals in range(6):
            allowed.add(round(v, decimals))
    return allowed


def detect_hallucinated_numbers(ai_text: str, rbm: RuleBasedMetrics) -> list[str]:
    """
    Scan the NUMBERS: lines in the AI output for values not present in the
    injected DSS metrics.

    Parameters
    ----------
    ai_text : str             — raw LLM response text.
    rbm     : RuleBasedMetrics — the same metrics object injected into the prompt.

    Returns
    -------
    list[str]
        Human-readable descriptions of suspicious numbers with their line
        context.  Empty list → all numbers are traceable to the injected metrics.

    Algorithm
    ---------
    1. Find every ``NUMBERS:`` line in the output.
    2. Extract all numeric tokens (regex ``\\b\\d+\\.?\\d*\\b``).
    3. Flag any token not within 2% of any allowed value (handles rounding artefacts).
    """
    allowed = _build_allowed_numbers(rbm)
    suspicious: list[str] = []

    numbers_lines = re.findall(r"NUMBERS:\s*(.+?)(?:\n|$)", ai_text, re.IGNORECASE)

    for line in numbers_lines:
        tokens = re.findall(r"\b\d+\.?\d*\b", line)
        for tok in tokens:
            try:
                val = float(tok)
                if val <= 0:
                    continue  # zeros are never an issue
                is_close = any(
                    abs(val - a) <= max(abs(a) * 0.02, 0.001)
                    for a in allowed
                    if a > 0
                )
                if not is_close:
                    ctx = line.strip()[:70]
                    entry = f"{tok}  ←  in: \"{ctx}\""
                    if entry not in suspicious:
                        suspicious.append(entry)
            except ValueError:
                pass

    return suspicious


# ---------------------------------------------------------------------------
# DSS Integrity: number sanitiser (safety net)
# ---------------------------------------------------------------------------

def sanitize_numbers_lines(ai_text: str, rbm: RuleBasedMetrics) -> str:
    """
    Safety-net post-processor: replace every NUMBERS: line in the LLM output
    with the correct Python-computed string, regardless of what the model wrote.

    This is the second layer of defence after the prompt already pre-writes
    the NUMBERS values.  Even if gemma3:1b paraphrases or garbles a NUMBERS
    line, this function restores the correct value deterministically.

    Slot assignment (same logic as build_dss_prompt):
      If vehicle_is_already_optimal (Train/Bus/e-tuk/Bicycle):
        Slot 1 → occupancy numbers
        Slot 2 → occupancy numbers
        Slot 3 → occupancy numbers
      Else if already greener than train baseline:
        Slot 1 → occupancy numbers
        Slot 2 → occupancy numbers
        Slot 3 → occupancy numbers
      Else (high-emission vehicle):
        Slot 1 → train numbers
        Slot 2 → occupancy numbers
        Slot 3 → train numbers
    """
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

    # Match the logic in build_dss_prompt
    if rbm.vehicle_is_already_optimal:
        slot_nums = [_n_occ, _n_occ, _n_occ]  # All occupancy-focused
    elif rbm.current_is_greener_than_train:
        slot_nums = [_n_occ, _n_occ, _n_occ]  # All occupancy-focused
    else:
        slot_nums = [_n_train, _n_occ, _n_train]  # Mixed: mode-switch + occupancy

    slot = 0
    result = []
    for line in ai_text.split("\n"):
        if re.match(r"\s*NUMBERS:", line, re.IGNORECASE) and slot < 3:
            indent = re.match(r"(\s*)", line).group(1)
            result.append(f"{indent}NUMBERS: {slot_nums[slot]}")
            slot += 1
        else:
            result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------

_STATIC_FALLBACK: dict[str, str] = {
    "low": (
        "🌱 Lovely work — your trip has a **LOW** carbon footprint!\n\n"
        "1. Keep hopping on Sri Lanka's scenic hill-country trains (Kandy–Ella) — "
        "they emit ~80% less CO₂ than a private car over the same route.\n"
        "2. Continue choosing eco-lodges or solar-powered guesthouses; "
        "Nuwara Eliya and Arugam Bay have great options.\n"
        "3. Stick to local rice-and-curry spots — they use fresh, low-emission "
        "produce and almost zero packaging waste.\n\n"
        "You’re already a role model for sustainable travel in Sri Lanka — keep it up! 🤝"
    ),
    "medium": (
        "⚠️ Your trip is at a **MEDIUM** emission level — a few tweaks go a long way!\n\n"
        "1. Swap your private vehicle for a shared tuk-tuk or an e-tuk (available via "
        "PickMe in Colombo and Kandy) for trips under 10 km.\n"
        "2. Ask your accommodation to turn off the diesel generator at night — "
        "most guesthouses in Ella and Galle now offer solar power as an alternative.\n"
        "3. Choose two plant-based 'rice and curry' meals per day; "
        "Sri Lankan vegetarian cuisine is delicious and cuts food emissions by ~25%.\n\n"
        "Small changes, big impact — you’ve got this! 🌊"
    ),
    "high": (
        "🔴 Your trip has a **HIGH** carbon footprint — here’s how to turn it around!\n\n"
        "1. Take the Ella Odyssey or Coastal Line train instead of a private car — "
        "it’s cheaper, stunning, and slashes your transport emissions dramatically.\n"
        "2. Switch to an eco-lodge with solar panels; generator-heavy accommodation "
        "is often your biggest emission source — ask before you book.\n"
        "3. Join a shared jeep safari (6 passengers) at Yala or Minneriya instead of "
        "hiring solo — you’ll cut per-person transport emissions by up to 80%.\n\n"
        "Sri Lanka’s green alternatives are world-class — give them a go next time! 🐘"
    ),
}


def _static_fallback(emission_level: str) -> str:
    return _STATIC_FALLBACK.get(
        emission_level.lower(),
        "No recommendation available. Please check the emission level value."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ollama_recommendation(
    emission_level: str,
    trip_data: dict,
    location: str = "Sri Lanka",
    vehicle_type: str = "Private Car",
    metrics: Optional[TransportMetrics] = None,
    rbm: Optional[RuleBasedMetrics] = None,
    model: str = "gemma3:1b",
) -> tuple[str, bool]:
    """
    Generate a quantitative, Sri Lanka-specific transport recommendation.

    Parameters
    ----------
    emission_level : str
        The ML model's classification: 'low', 'medium', or 'high'.
    trip_data : dict
        Dictionary of trip input features (same keys as the Streamlit inputs).
    location : str
        The tourist's current Sri Lankan destination (e.g. 'Ella', 'Colombo').
    vehicle_type : str
        The primary vehicle used (e.g. 'Private Car', 'Train').
    metrics : Optional[TransportMetrics]
        Legacy quantitative metrics (used as fallback if rbm is None).
    rbm : Optional[RuleBasedMetrics]
        Level-2 DSS rule-based metrics from ``run_rule_based_analysis``.
        When provided, ``build_dss_prompt`` is used for the highest-quality output.
    model : str
        Ollama model tag. Default: 'gemma3:1b'.

    Returns
    -------
    (recommendation_text, ai_generated) : tuple[str, bool]
        ai_generated is True if Ollama responded, False if static fallback was used.
    """
    # Prefer the Level-2 DSS prompt when RuleBasedMetrics are available
    if rbm is not None:
        prompt = build_dss_prompt(rbm, location)
    else:
        prompt = _build_quantitative_prompt(emission_level, trip_data, location, vehicle_type, metrics)

    try:
        # Use ollama.generate() with the /api/generate endpoint
        response = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False,
        )
        text = response["response"].strip()
        return text, True

    except Exception as exc:  # noqa: BLE001
        fallback_text = _static_fallback(emission_level)
        return (
            f"{fallback_text}\n\n"
            f"_(AI unavailable: {type(exc).__name__} — "
            f"ensure Ollama is running and `{model}` is pulled.)_"
        ), False
