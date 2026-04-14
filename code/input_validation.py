"""
input_validation.py
───────────────────────────────────────────────────────────────────────────────
Layer 0: Input Validation — runs BEFORE the ML model prediction.

Catches physically impossible or logically unrealistic input combinations
so that downstream layers (ML, rule-based engine, LLM) always receive
clean, credible data.

Checks performed
────────────────
  1. Vehicle capacity    — passengers cannot exceed physical seat count
  2. Distance sanity     — zero/negative distance is blocked
  3. Sri Lanka geography — warns if distance exceeds the island's practical limit
  4. Trip-day realism    — distance/day ratio check for Sri Lankan roads
  5. Emission physics    — transport_emissions_kgCO2 floor based on vehicle type
  6. Negative values     — any emission field < 0 is blocked
  7. Bicycle/Motorcycle  — hard per-vehicle passenger limits
"""

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Physical constraints
# ─────────────────────────────────────────────────────────────────────────────

# Maximum seated passenger capacity per vehicle type (Sri Lanka context).
# Sources: vehicle manufacturers / Sri Lanka Transport Board regulations.
VEHICLE_CAPACITY: dict[str, int] = {
    "Private Car":               5,   # incl. driver
    "Tourist Van / Minibus":    14,   # Toyota HiAce / Nissan Caravan typical
    "Tuk-tuk (Three-wheeler)":   3,   # driver + 2 passengers
    "Electric Tuk-tuk (e-tuk)":  3,   # same bodywork as petrol tuk-tuk
    "Public Bus (CTB)":         60,   # inter-city CTB bus seated capacity
    "Train":                   500,   # upper bound for a single trip group
    "Motorcycle":                2,   # rider + pillion
    "Bicycle":                   1,   # rider only
}

# Minimum realistic vehicle-level emission rate (kgCO₂ per vehicle-km).
# Values are conservative lower bounds (highly efficient vehicles, ideal
# conditions).  Used as the physics floor for auto-correction in Rule 5.
#
# Citations
# ─────────
# Private Car (0.060 kg/km floor = 60 g/km minimum):
#   UK DESNZ (2023) "Greenhouse Gas Reporting: Conversion Factors", Table 1;
#   super-efficient petrol or mild-hybrid — the most favourable real-world
#   figure. Average UK medium petrol car is ~170 g/km.
# Train       (0.300 kg/km floor = 300 g/km locomotive total):
#   IEA (2023) "CO₂ Emissions from Fuel Combustion", rail fuel consumption
#   data; conservative floor for a diesel locomotive under light load.
MIN_VEHICLE_EF_KG_PER_KM: dict[str, float] = {
    "Private Car":               0.060,   # 60 g/km  — DESNZ 2023 super-efficient petrol floor
    "Tourist Van / Minibus":     0.100,   # 100 g/km — efficient diesel van
    "Tuk-tuk (Three-wheeler)":   0.040,   # 40 g/km  — 2/4-stroke
    "Electric Tuk-tuk (e-tuk)":  0.005,   # 5 g/km   — Sri Lanka grid 2024
    "Public Bus (CTB)":          0.180,   # 180 g/km — vehicle total
    "Train":                     0.300,   # 300 g/km — IEA 2023 diesel locomotive
    "Motorcycle":                0.040,   # 40 g/km
    "Bicycle":                   0.000,   # zero operational emissions
}

# Longest practical road/rail route within Sri Lanka (Colombo → Jaffna ≈ 400 km)
MAX_REALISTIC_DISTANCE_KM: float = 450.0


# ─────────────────────────────────────────────────────────────────────────────
# Severity constants
# ─────────────────────────────────────────────────────────────────────────────

ERROR   = "error"     # Blocks ML prediction — user MUST fix before proceeding
WARNING = "warning"   # Advisory only — prediction is allowed but user should review


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """One validation finding (error or warning)."""
    severity: str   # ERROR | WARNING
    field:    str   # which UI field caused the issue
    message:  str   # human-readable explanation shown in the Streamlit UI


@dataclass
class ValidationResult:
    """Aggregated result of calling validate_inputs()."""
    issues: list[ValidationIssue] = field(default_factory=list)
    # Physics-corrected replacement values applied when inputs fall below the
    # realism floor.  Keys match trip_data field names; values are floats.
    # The caller (app.py) applies these to input_dict before ML inference.
    auto_corrections: dict[str, float] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        """True when at least one blocking error was found."""
        return any(i.severity == ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """True when at least one non-blocking warning was found."""
        return any(i.severity == WARNING for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def is_clean(self) -> bool:
        """True when no issues were found at all."""
        return len(self.issues) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_inputs(trip_data: dict, vehicle_type: str) -> ValidationResult:
    """
    Validate trip inputs for physical and logical realism.

    Parameters
    ----------
    trip_data    : dict — same keys/values as the Streamlit input_dict.
    vehicle_type : str  — exact string from the vehicle selectbox.

    Returns
    -------
    ValidationResult
        .has_errors   → True  — block prediction; show errors to user.
        .has_warnings → True  — show advisory notices; allow prediction.
        .is_clean     → True  — all checks passed; no issues detected.

    Validation Rules
    ----------------
    Rule 1  Vehicle capacity     — occupancy > VEHICLE_CAPACITY[vehicle]  → ERROR
    Rule 2  Distance positive    — distance_km ≤ 0                        → ERROR
    Rule 3  Sri Lanka geography  — distance_km > MAX_REALISTIC_DISTANCE   → WARNING
    Rule 4  Trip-day realism     — distance > 300 km in 1 day             → WARNING
    Rule 5  Emission physics     — transport_kgCO2 < 10% of physics floor → WARNING
    Rule 6  Negative values      — any emission field < 0                 → ERROR
    Rule 7  Motorcycle passengers — occupancy > 2                         → ERROR
    """
    result = ValidationResult()

    # ── Extract fields with safe defaults ────────────────────────────────────
    occupancy       = int(float(trip_data.get("occupancy",                 1)))
    distance_km     = float(trip_data.get("distance_km",                   1.0))
    transport_kgco2 = float(trip_data.get("transport_emissions_kgCO2",     0.0))
    trip_days       = int(float(trip_data.get("trip_days",                 1)))
    food_kgco2      = float(trip_data.get("food_emissions_kgCO2",          0.0))
    waste_kgco2     = float(trip_data.get("waste_emissions_kgCO2",         0.0))
    plastic_kgco2   = float(trip_data.get("plastic_emissions_kgCO2",       0.0))
    accom_elec      = float(trip_data.get("accommodation_elec_kgCO2",      0.0))
    accom_gen       = float(trip_data.get("accommodation_gen_kgCO2",       0.0))

    # ── Rule 1: Vehicle capacity ──────────────────────────────────────────────
    max_cap = VEHICLE_CAPACITY.get(vehicle_type, 999)
    if occupancy > max_cap:
        result.issues.append(ValidationIssue(
            severity=ERROR,
            field="occupancy",
            message=(
                f"**{vehicle_type}** seats a maximum of **{max_cap} passenger(s)**. "
                f"You entered **{occupancy}**. "
                "Reduce the passenger count or select a larger vehicle "
                "(e.g. Tourist Van / Minibus or Public Bus)."
            ),
        ))

    # ── Rule 2: Distance must be positive ────────────────────────────────────
    if distance_km <= 0:
        result.issues.append(ValidationIssue(
            severity=ERROR,
            field="distance_km",
            message="Travel distance must be greater than 0 km.",
        ))

    # ── Rule 3: Distance vs Sri Lanka geography ───────────────────────────────
    if distance_km > MAX_REALISTIC_DISTANCE_KM:
        result.issues.append(ValidationIssue(
            severity=WARNING,
            field="distance_km",
            message=(
                f"Distance of **{distance_km:.0f} km** exceeds the longest practical "
                f"road route in Sri Lanka (~{MAX_REALISTIC_DISTANCE_KM:.0f} km). "
                "If this includes an international leg (e.g. a flight), consider splitting "
                "transport legs or verifying the distance."
            ),
        ))

    # ── Rule 4: Trip days vs distance realism ─────────────────────────────────
    # Average Sri Lankan road speed ~50 km/h; flag trips > 300 km in 1 day
    if distance_km > 300 and trip_days == 1:
        result.issues.append(ValidationIssue(
            severity=WARNING,
            field="trip_days",
            message=(
                f"A **{distance_km:.0f} km** trip in **1 day** is unusually far for "
                "Sri Lankan roads (average driving speed ~50 km/h). "
                "Please verify the distance and trip duration."
            ),
        ))

    # ── Rule 5: Transport emission physics floor ──────────────────────────────
    # Expected minimum TOTAL vehicle emission = MIN_EF × distance_km.
    # Flag if entered value is < 10% of this floor (likely a data error).
    min_ef = MIN_VEHICLE_EF_KG_PER_KM.get(vehicle_type, 0.0)
    expected_floor = min_ef * distance_km   # kgCO₂ — minimum plausible vehicle total

    if (
        vehicle_type != "Bicycle"
        and min_ef > 0
        and distance_km > 5
        and transport_kgco2 < expected_floor * 0.1   # below 10% of physics floor
    ):
        # Auto-correct: replace the entered value with the physics floor so that
        # downstream per-passenger and g/pax-km metrics remain physically credible.
        corrected = round(expected_floor, 5)
        result.auto_corrections["transport_emissions_kgCO2"] = corrected
        result.issues.append(ValidationIssue(
            severity=WARNING,
            field="transport_emissions_kgCO2",
            message=(
                f"Transport emission of **{transport_kgco2:.5f} kgCO₂** for a "
                f"**{vehicle_type}** over **{distance_km:.0f} km** is below the physics "
                f"floor (~**{expected_floor:.3f} kgCO₂** minimum for this vehicle type). "
                f"⚙️ **Auto-corrected to {corrected:.5f} kgCO₂** for this prediction. "
                "Per-passenger and g/pax-km metrics will use the corrected value. "
                "To avoid this, recalculate transport emissions from actual fuel use "
                f"(e.g. {min_ef*1000:.0f} g/km × {distance_km:.0f} km = {expected_floor:.3f} kgCO₂)."
            ),
        ))

    # ── Rule 6: Negative emission values ─────────────────────────────────────
    neg_check = {
        "transport_emissions_kgCO2":  transport_kgco2,
        "food_emissions_kgCO2":       food_kgco2,
        "waste_emissions_kgCO2":      waste_kgco2,
        "plastic_emissions_kgCO2":    plastic_kgco2,
        "accommodation_elec_kgCO2":   accom_elec,
        "accommodation_gen_kgCO2":    accom_gen,
    }
    for field_name, value in neg_check.items():
        if value < 0:
            result.issues.append(ValidationIssue(
                severity=ERROR,
                field=field_name,
                message=f"**{field_name}** cannot be negative. Entered: {value:.4f}.",
            ))

    # ── Rule 7: Motorcycle passenger hard limit ───────────────────────────────
    # (Caught by Rule 1 already, but give a more specific message for Motorcycles)
    if vehicle_type == "Motorcycle" and occupancy > 2 and occupancy <= max_cap:
        result.issues.append(ValidationIssue(
            severity=ERROR,
            field="occupancy",
            message=(
                f"A **Motorcycle** can carry a maximum of **2 people** (rider + pillion). "
                f"You entered **{occupancy}**."
            ),
        ))

    return result
