import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from emission_analysis import compute_transport_metrics, run_rule_based_analysis, EMISSION_FACTORS_PKM
from ollama_recommendations import get_ollama_recommendation, parse_recommendation_output, detect_hallucinated_numbers, sanitize_numbers_lines
from input_validation import validate_inputs, ERROR, WARNING

# Load model and features
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
features_path = os.path.join(os.path.dirname(__file__), "../model/features.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

st.title("🌍 Tourism Carbon Footprint Calculator")
st.write("This app predicts the overall carbon emission level for a tourist's trip based on comprehensive travel data.")

# Create organized input sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚗 Trip Details")
    trip_days = st.number_input("Trip Duration (days)", 1, 30, 6)
    distance_km = st.number_input("Travel Distance (km)", 1.0, 1000.0, 54.8)
    occupancy = st.number_input("Vehicle Occupancy", 1, 100, 2)
    
    st.subheader("🌍 Environmental Factors")
    congestion_factor = st.slider("Congestion Factor (0.8-1.2)", 0.8, 1.2, 0.91, 0.01)
    terrain_factor = st.slider("Terrain Factor (1.0-1.3)", 1.0, 1.3, 1.15, 0.01)

with col2:
    st.subheader("🏨 Accommodation")
    nightly_kwh_est = st.number_input("Nightly Electricity (kWh)", 1.0, 200.0, 8.44)
    grid_ef_kgCO2_per_kWh = st.number_input("Grid Carbon Efficiency (kgCO2/kWh)", 0.01, 1.0, 0.07, 0.01)
    
    st.subheader("🍽️ Food & Waste")
    food_emissions_kgCO2 = st.number_input("Food Emissions (kgCO2)", 0.0, 100.0, 16.884)
    waste_emissions_kgCO2 = st.number_input("Waste Emissions (kgCO2)", 0.0, 50.0, 2.998)
    plastic_emissions_kgCO2 = st.number_input("Plastic Emissions (kgCO2)", 0.0, 10.0, 0.3)

col3, col4 = st.columns(2)

with col3:
    st.subheader("🚌 Transport Emissions")
    transport_emissions_kgCO2 = st.number_input("Transport Emissions (kgCO2)", 0.0, 500.0, 0.037)

with col4:
    st.subheader("⚡ Accommodation Emissions")
    accommodation_elec_kgCO2 = st.number_input("Electricity Emissions (kgCO2)", 0.0, 200.0, 4.805)
    accommodation_gen_kgCO2 = st.number_input("Generator Emissions (kgCO2)", 0.0, 200.0, 11.095)

# --- Sri Lanka context for AI recommendations ---
st.divider()
st.subheader("🗺️ Trip Context (for AI Recommendations)")
ctx_col1, ctx_col2 = st.columns(2)

with ctx_col1:
    location = st.selectbox(
        "📍 Location in Sri Lanka",
        [
            "Colombo", "Kandy", "Ella", "Galle", "Nuwara Eliya",
            "Sigiriya", "Trincomalee", "Yala", "Arugam Bay", "Anuradhapura",
        ],
        index=2,  # default: Ella
        help="The AI will tailor recommendations to this specific Sri Lankan destination.",
    )

with ctx_col2:
    vehicle_type = st.selectbox(
        "🚗 Primary Vehicle Used",
        [
            "Private Car",
            "Tourist Van / Minibus",
            "Tuk-tuk (Three-wheeler)",
            "Electric Tuk-tuk (e-tuk)",
            "Public Bus (CTB)",
            "Train",
            "Motorcycle",
            "Bicycle",
        ],
        index=0,
        help="The AI uses this to suggest greener local alternatives.",
    )

# Quick static badge shown instantly while AI recommendation loads
def get_quick_badge(level: str) -> str:
    badges = {
        "low":    "✅ **Low** — your trip has a low carbon footprint. See the AI recommendation below for tips to stay green.",
        "medium": "⚠️ **Medium** — there is room to improve. See the AI recommendation below for targeted actions.",
        "high":   "🔴 **High** — significant emissions detected. See the AI recommendation below for urgent actions.",
    }
    return badges.get(level.lower(), "Emission level classified. Check the AI recommendation below.")

def align_features(input_df, expected_columns):
    df_local = pd.get_dummies(input_df)
    for col in expected_columns:
        if col not in df_local.columns:
            df_local[col] = 0
    df_local = df_local[expected_columns]
    return df_local

# Prediction Button
if st.button("🔮 Predict Emission Level", use_container_width=True):
    # Prepare input data in correct order
    input_dict = {
        'trip_days': trip_days,
        'distance_km': distance_km,
        'occupancy': occupancy,
        'congestion_factor': congestion_factor,
        'terrain_factor': terrain_factor,
        'nightly_kwh_est': nightly_kwh_est,
        'grid_ef_kgCO2_per_kWh': grid_ef_kgCO2_per_kWh,
        'food_emissions_kgCO2': food_emissions_kgCO2,
        'waste_emissions_kgCO2': waste_emissions_kgCO2,
        'plastic_emissions_kgCO2': plastic_emissions_kgCO2,
        'transport_emissions_kgCO2': transport_emissions_kgCO2,
        'accommodation_elec_kgCO2': accommodation_elec_kgCO2,
        'accommodation_gen_kgCO2': accommodation_gen_kgCO2
    }

    # ── Layer 0: Input Validation ──────────────────────────────────────────────
    # Runs before ML inference. Blocks physically impossible inputs (e.g. 27
    # passengers in a Private Car) and warns about suspicious emission values.
    vr = validate_inputs(input_dict, vehicle_type)

    if vr.has_errors:
        st.divider()
        st.error("🚫 **Input validation failed** — fix the errors below before predicting.")
        for issue in vr.errors:
            st.error(f"**{issue.field}:** {issue.message}")
        if vr.has_warnings:
            with st.expander("⚠️ Additional warnings", expanded=False):
                for issue in vr.warnings:
                    st.warning(issue.message)
        st.stop()

    if vr.has_warnings:
        st.divider()
        with st.expander("⚠️ Input warnings — prediction will proceed", expanded=True):
            for issue in vr.warnings:
                st.warning(f"**{issue.field}:** {issue.message}")

    # Apply physics-based auto-corrections before ML inference and rule-based analysis.
    # The corrected values are already described in the warnings shown above.
    if vr.auto_corrections:
        for _corr_field, _corr_val in vr.auto_corrections.items():
            input_dict[_corr_field] = _corr_val

    input_df = pd.DataFrame([input_dict])
    aligned_df = align_features(input_df, feature_columns)
    # Pass DataFrame directly (not .values) so the model receives the feature
    # names it was trained with — eliminates sklearn UserWarning.
    prediction = model.predict(aligned_df)[0]

    # ── Layer 2.5: ML Verification & Correction ──────────────────────────────────
    # The ML model sometimes misclassifies based on individual feature patterns.
    # This verification layer checks against the actual total emissions and corrects
    # obvious misclassifications using the empirical thresholds from training data.
    #
    # Thresholds (from 5000-record training dataset, percentile-based):
    #   LOW:    ≤ 67.25 kgCO₂
    #   MEDIUM: 67.25 - 162.92 kgCO₂
    #   HIGH:   > 162.92 kgCO₂
    
    actual_total_emissions = (
        transport_emissions_kgCO2 + accommodation_elec_kgCO2 + 
        accommodation_gen_kgCO2 + food_emissions_kgCO2 + 
        waste_emissions_kgCO2 + plastic_emissions_kgCO2
    )
    
    # Define thresholds (same as training.py)
    THRESHOLD_LOW_TO_MEDIUM = 67.25
    THRESHOLD_MEDIUM_TO_HIGH = 162.92
    
    def classify_by_actual_emissions(total_emissions):
        """Classify based on actual total emissions (ground truth)."""
        if total_emissions <= THRESHOLD_LOW_TO_MEDIUM:
            return 'low'
        elif total_emissions <= THRESHOLD_MEDIUM_TO_HIGH:
            return 'medium'
        else:
            return 'high'
    
    actual_classification = classify_by_actual_emissions(actual_total_emissions)
    
    # Compare ML prediction vs actual total emissions
    ml_confidence_issue = False
    if prediction.lower() != actual_classification:
        ml_confidence_issue = True
        # Override the prediction with the ground-truth classification
        original_ml_prediction = prediction
        prediction = actual_classification
    
    # ── ML Emission level badge ──────────────────────────────────────────────────
    st.divider()
    level_colours = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    icon = level_colours.get(prediction.lower(), "⚪")
    st.subheader(f"{icon} ML Classification: Emission Level **{prediction.upper()}**")
    
    # Show warning if ML prediction was corrected
    if ml_confidence_issue:
        st.warning(
            f"⚠️ **ML Verification Applied**: Actual total emissions "
            f"({actual_total_emissions:.2f} kgCO₂) indicate **{actual_classification.upper()}** level, "
            f"not {original_ml_prediction.upper()}. Classification corrected to match ground truth."
        )
    
    st.info(get_quick_badge(prediction))

    # ── Level-2 DSS: Rule-Based Layer ─────────────────────────────────────────
    rbm = run_rule_based_analysis(input_dict, vehicle_type, prediction)

    st.divider()
    st.subheader("📂 Level-2 DSS: Rule-Based Transport Metrics")
    st.caption(
        "🔢 Four metrics computed by a **Python rule engine** — "
        "not the ML model — and injected verbatim into the Ollama prompt."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Metric 1 · Per-Passenger Emission",
            f"{rbm.per_passenger_kgCO2:.5f} kgCO₂",
            help=(
                f"{rbm.total_transport_kgCO2:.5f} kgCO₂ ÷ {rbm.current_passengers} pax "
                f"= {rbm.per_passenger_g_per_km:.1f} g/pax-km"
            ),
        )
    with m2:
        occ_label = (
            f"Metric 2 · Full Occ ({rbm.full_occ_passengers} pax)"
            if rbm.current_passengers < rbm.full_occ_passengers
            else "Metric 2 · Already at Full Occ"
        )
        st.metric(
            occ_label,
            f"{rbm.full_occ_per_passenger_kgCO2:.5f} kgCO₂/pax",
            delta=f"-{rbm.full_occ_reduction_pct}%" if rbm.full_occ_reduction_pct > 0 else "—",
            delta_color="inverse",
            help=(
                f"Rate at full occupancy: **{rbm.full_occ_g_per_km:.1f} g/pax-km** "
                f"(vs {rbm.per_passenger_g_per_km:.1f} g/pax-km now). "
                f"Filling all {rbm.full_occ_passengers} seats saves "
                f"{rbm.full_occ_reduction_pct}% per passenger."
            ),
        )
    with m3:
        train_delta = (
            f"-{rbm.vs_train_saving_pct}%"
            if not rbm.current_is_greener_than_train
            else "✅ Already greener"
        )
        st.metric(
            "Metric 3 · Train Baseline (14 g/pax-km)",
            f"{rbm.train_baseline_kgCO2:.5f} kgCO₂/pax",
            delta=train_delta,
            delta_color="inverse" if not rbm.current_is_greener_than_train else "off",
            help=(
                f"Rate comparison: **{rbm.per_passenger_g_per_km:.1f} g/pax-km** (current) "
                f"vs **14.0 g/pax-km** (IEA 2023 train baseline). "
                f"Source: IEA (2023) CO₂ Emissions from Fuel Combustion, p. 185; "
                f"Our World in Data (Ritchie 2020)."
            ),
        )
    with m4:
        eff_icon = {"EFFICIENT": "🟢", "MODERATE": "🟡", "INEFFICIENT": "🔴"}.get(rbm.efficiency_label, "⚪")
        st.metric(
            "Metric 4 · Efficiency Rating",
            f"{eff_icon} {rbm.efficiency_label}",
            help=rbm.efficiency_note,
        )

    # ── 3-bar comparison chart: current / full-occ / train baseline ──────────
    # Chart uses g CO₂/pax-km (rate) so all three bars are on the same scale
    # regardless of trip distance — enabling direct cross-mode comparison.
    st.subheader("💨 Emission Comparison")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(f"🔵 Current ({vehicle_type})", f"{rbm.per_passenger_g_per_km:.1f}", "g CO₂/pax-km")
    with col_m2:
        st.metric(f"🟡 Full Occ ({rbm.full_occ_passengers} pax)", f"{rbm.full_occ_g_per_km:.1f}", "g CO₂/pax-km")
    with col_m3:
        st.metric("🟢 Train Baseline", f"{rbm.train_baseline_g_per_pkm:.1f}", "g CO₂/pax-km")
    st.caption(
        "📊 Y-axis: **g CO₂ / pax-km** — a distance-normalised, per-person rate that makes "
        "vehicles of different sizes directly comparable. "
        "Train baseline: **14.0 g/pax-km** "
        "(IEA 2023 *CO₂ Emissions from Fuel Combustion*, p. 185; "
        "Our World in Data, Ritchie 2020)."
    )

    # ── AI-powered quantitative recommendation via Ollama ─────────────────────
    st.divider()
    st.subheader("🤖 AI Transport Advisory (Ollama · gemma3:1b)")
    st.caption(f"📍 Location: **{location}**  |  🚗 Vehicle: **{vehicle_type}**  |  📈 Metrics injected into prompt: ✅")

    with st.spinner("Generating quantitative transport recommendations — please wait…"):
        ai_text, ai_ok = get_ollama_recommendation(
            emission_level=prediction,
            trip_data=input_dict,
            location=location,
            vehicle_type=vehicle_type,
            rbm=rbm,
        )

    if ai_ok:
        st.success("✅ Quantitative recommendations generated by Ollama (gemma3:1b)")
        # ── Safety net: overwrite any NUMBERS lines with Python-computed values ──
        ai_text = sanitize_numbers_lines(ai_text, rbm)
        # ── DSS Integrity Check: scan AI output for hallucinated numbers ──────
        suspicious = detect_hallucinated_numbers(ai_text, rbm)
        if suspicious:
            with st.expander(
                f"🔍 DSS Integrity Check — {len(suspicious)} unverified number(s) detected",
                expanded=False,
            ):
                st.warning(
                    "The AI output contains number(s) **not found in the injected DSS metrics**. "
                    "Cross-check the NUMBERS lines against the DSS table below "
                    "before quoting these figures."
                )
                for s in suspicious:
                    st.code(s, language=None)
    else:
        st.warning("⚠️ Ollama unavailable — showing static fallback recommendation.")

    # Try to display as structured cards; fall back to raw markdown
    parsed = parse_recommendation_output(ai_text)

    if parsed and parsed.get("recommendations"):
        for i, rec in enumerate(parsed["recommendations"], 1):
            with st.container(border=True):
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(f"**{i}. {rec['action']}**")
                    if rec["tip"]:
                        st.caption(rec["tip"])
                with right:
                    if rec["numbers"]:
                        st.code(rec["numbers"], language=None)
        if parsed.get("verdict"):
            st.info(f"🎯 **Verdict:** {parsed['verdict']}")
    else:
        # Fallback: raw LLM output (parser couldn’t find the expected format)
        with st.container(border=True):
            st.markdown(ai_text)

    # ── Expandable: DSS metrics table + full breakdown ───────────────────────
    with st.expander("🔢 DSS Metrics injected into the Ollama prompt", expanded=False):
        st.markdown("**Rule-Based Layer — 4 pre-computed metrics sent verbatim to the LLM**")
        dss_table = pd.DataFrame([
            {
                "Metric":   "1 · Per-Passenger Emission",
                "Value":    f"{rbm.per_passenger_kgCO2:.5f} kgCO₂/pax",
                "Detail":   (
                    f"{rbm.total_transport_kgCO2:.5f} kgCO₂ ÷ {rbm.current_passengers} pax"
                    f" = {rbm.per_passenger_g_per_km:.1f} g/pax-km"
                ),
            },
            {
                "Metric":   f"2 · Full-Occupancy Scenario ({rbm.full_occ_passengers} pax)",
                "Value":    f"{rbm.full_occ_per_passenger_kgCO2:.5f} kgCO₂/pax",
                "Detail":   (
                    f"{rbm.current_passengers} → {rbm.full_occ_passengers} pax  →  "
                    f"save {rbm.full_occ_reduction_pct}%"
                ),
            },
            {
                "Metric":   "3 · Train Baseline (14 g/pax-km)",
                "Value":    f"{rbm.train_baseline_kgCO2:.5f} kgCO₂/pax",
                "Detail":   (
                    f"Current {rbm.per_passenger_g_per_km:.1f} g/pax-km vs 14 g/pax-km  →  "
                    + (
                        f"save {rbm.vs_train_saving_pct}% by switching to train"
                        if not rbm.current_is_greener_than_train
                        else "current vehicle is already greener ✅"
                    )
                ),
            },
            {
                "Metric":   "4 · Efficiency Rating",
                "Value":    rbm.efficiency_label,
                "Detail":   rbm.efficiency_note,
            },
        ])
        # Display as formatted text instead of dataframe to avoid PyArrow
        for idx, row in dss_table.iterrows():
            st.markdown(f"**{row['Metric']}**")
            st.markdown(f"- Value: {row['Value']}")
            st.markdown(f"- Detail: {row['Detail']}")

        st.markdown("**Full emission breakdown (all categories)**")
        breakdown_df = pd.DataFrame([
            {"Category": "Food",                        "kgCO₂": input_dict["food_emissions_kgCO2"]},
            {"Category": "Transport",                   "kgCO₂": input_dict["transport_emissions_kgCO2"]},
            {"Category": "Accommodation (electricity)", "kgCO₂": input_dict["accommodation_elec_kgCO2"]},
            {"Category": "Accommodation (generator)",  "kgCO₂": input_dict["accommodation_gen_kgCO2"]},
            {"Category": "Waste",                      "kgCO₂": input_dict["waste_emissions_kgCO2"]},
            {"Category": "Plastic",                    "kgCO₂": input_dict["plastic_emissions_kgCO2"]},
        ])
        # Display as formatted text instead of dataframe to avoid PyArrow
        breakdown_sorted = breakdown_df.sort_values("kgCO₂", ascending=False).reset_index(drop=True)
        for idx, row in breakdown_sorted.iterrows():
            st.markdown(f"**{row['Category']}**: {row['kgCO₂']:.3f} kgCO₂")

