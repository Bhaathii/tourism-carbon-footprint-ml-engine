# 5 LAYERS OF YOUR SYSTEM - FILE MAPPING

## **LAYER 1: INPUT VALIDATION** 🚫
**File:** [code/input_validation.py](code/input_validation.py)

**What it does:**
- Checks if user inputs make physical sense
- Prevents impossible combinations (e.g., Private Car with 100 passengers)
- Auto-corrects unrealistic values

**Key Code:**
```python
VEHICLE_CAPACITY: dict[str, int] = {
    "Private Car": 5,
    "Tourist Van / Minibus": 14,
    "Tuk-tuk (Three-wheeler)": 3,
    "Public Bus (CTB)": 60,
    "Train": 500,
}

# Minimum emission physics floor (kgCO2/km)
VEHICLE_EMISSION_FLOOR: dict[str, float] = {
    "Private Car": 0.060,  # 60 g/km minimum
    "Motorcycle": 0.045,
    "Bicycle": 0.000,
}
```

**Checks Performed:**
1. ✓ Vehicle capacity (passengers ≤ seat count)
2. ✓ Distance sanity (not negative, not zero)
3. ✓ Sri Lanka geography (55 km is reasonable for an island)
4. ✓ Emission physics (transport_emissions_kgCO2 must exceed vehicle floor)
5. ✓ Negative values (all emissions ≥ 0)

**Real Example from Your Run:**
```
User input: transport_emissions = 0.037 kgCO2
Distance: 55 km, Vehicle: Private Car

LAYER 1 detects:
  "0.037 kgCO2 is below physics floor (3.288 kgCO2)"
  
Action: Auto-correct to 3.288 kgCO2 ✓
```

---

## **LAYER 2: ML MODEL PREDICTION** 🧠
**Files:**
- [code/training.py](code/training.py) — Creates the ML model
- [code/app.py](code/app.py) — Uses the ML model

**What it does:**
- Trains Random Forest on 5,000 tourism records
- Classifies current trip as LOW / MEDIUM / HIGH emissions

**Key Code (Training):**
```python
# training.py
model = RandomForestClassifier(
    n_estimators=800,
    random_state=42,
    max_depth=20,
    class_weight='balanced',
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(feature_columns, "model/features.pkl")

# Performance
print("Accuracy:", accuracy_score(y_test, pred))  # 94.2%
```

**Key Code (Usage in app.py):**
```python
# app.py
model = joblib.load("model/model.pkl")
feature_columns = joblib.load("model/features.pkl")

# Prepare input in correct feature order
input_df = pd.DataFrame([input_dict])
input_df = align_features(input_df, feature_columns)

# Predict
prediction = model.predict(input_df)
# Output: ['medium']
```

**Real Example from Your Run:**
```
Input: 13 parameters (validated)
         ↓
Random Forest processes
         ↓
Output: "MEDIUM" (with 94.2% accuracy)
```

---

## **LAYER 3: PYTHON DSS CALCULATIONS** 📊
**File:** [code/emission_analysis.py](code/emission_analysis.py)

**What it does:**
- Calculates exact per-passenger emissions (NOT from AI)
- Computes alternative vehicle options and savings
- Generates 4 metrics to inject into Ollama prompt

**Key Code:**
```python
# emission_analysis.py
EMISSION_FACTORS_PKM: dict[str, float] = {
    "Private Car": 0.171,           # 171 g/pax-km
    "Tourist Van / Minibus": 0.052,
    "Public Bus (CTB)": 0.039,
    "Train": 0.035,                 # 35 g/pax-km (IEA baseline)
    "Electric Tuk-tuk (e-tuk)": 0.018,
    "Bicycle": 0.000,
}

class TransportMetrics:
    distance_km: float
    occupancy: int
    per_passenger_kgCO2: float  # <= Calculated here
    alternatives: list[Alternative]  # <- All modes compared

# Calculation:
per_passenger = total_emissions / occupancy
# 3.288 kgCO2 / 2 passengers = 1.644 kgCO2/pax
```

**Real Example from Your Run:**
```
Input (from validated Layer 1):
  distance = 55 km
  occupancy = 2 passengers
  vehicle = Private Car
  total_emissions = 3.288 kgCO2

Metric 1: Per-passenger = 3.288 / 2 = 1.644 kgCO2
Metric 2: Full occupancy (4 pax) = 0.822 kgCO2/pax → SAVE 50%
Metric 3: Train alternative = 0.767 kgCO2/pax → SAVE 53.3%
Metric 4: Rating = MODERATE

These 4 numbers are now injected into Ollama prompt
```

**Why This Layer is Critical:**
- Numbers are 100% calculated, not guessed
- No AI involved (AI can't do reliable math)
- Traceable to physics-based emission factors

---

## **LAYER 4: OLLAMA/GEMMA EXPLANATION** 💬
**File:** [code/ollama_recommendations.py](code/ollama_recommendations.py)

**What it does:**
- Receives metrics from Layer 3
- Sends them to Ollama/Gemma with instructions
- Gemma generates 3 human-readable recommendations

**Key Code:**
```python
# ollama_recommendations.py

def _build_quantitative_prompt(emission_level, trip_data, location, vehicle_type, metrics):
    """Build prompt injecting Layer 3 metrics."""
    
    prompt = f"""
    Trip Profile:
      Location: {location}  ← Ella, Colombo, etc.
      Vehicle: {vehicle_type}  ← Private Car, Train, etc.
      Distance: {metrics.distance_km} km
      
    Current Transport Emissions:
      Per passenger: {metrics.per_passenger_kgCO2:.5f} kgCO₂  ← FROM LAYER 3
      
    Alternatives (same route):
      Train: {metrics.train_baseline_kgCO2:.5f} kgCO₂ → save {metrics.vs_train_saving_pct}%
      Double occupancy: → save {metrics.reduction_pct_double_occupancy}%
      
    Local context for {location}:
      {location_context}  ← Hard-coded Sri Lankan knowledge
      
    INSTRUCTION: Write exactly 3 numbered recommendations using ONLY numbers above.
    Return format:
    1. ACTION: [name]
       NUMBERS: [value] → [value] → save [X]%
       TIP: [1-2 sentences with specific Sri Lankan service]
    """
    
    # Call Ollama
    response = ollama.generate(
        model="gemma3:1b",
        prompt=prompt,
        stream=False
    )
    
    return response["response"]
```

**Real Example from Your Run:**
```
Layer 3 Metrics Injected:
  Current: 30.0 g/pax-km
  Train: 14.0 g/pax-km
  Savings: 53.3%
  Location: Ella

Gemma Generates:
  "1. Switch to public transport
      Utilize the CTB bus for comfortable travel to Ella.
      30.0 g/pax-km → 14.0 g/pax-km → save 53.3%
   
   2. Share vehicle
      Share the private car with a companion exploring Ella.
      30.0 g/pax-km → 15.0 g/pax-km → save 50.0%
   
   3. Use e-tuk instead of car
      Opt for shared e-tuk service for eco-friendly travel.
      30.0 g/pax-km → 14.0 g/pax-km → save 53.3%"
```

**Key Control Mechanisms:**
- ✓ Ollama is told: "Use ONLY these numbers"
- ✓ Location-specific context provided (not AI guessing)
- ✓ Constrained output format (1. ACTION / NUMBERS / TIP)

---

## **LAYER 5: VERIFICATION & HALLUCINATION DETECTION** ✅
**File:** [code/ollama_recommendations.py](code/ollama_recommendations.py)

**What it does:**
- Scans every number Gemma wrote
- Checks if it came from Layer 3 metrics
- Replaces hallucinated numbers automatically

**Key Code:**
```python
# ollama_recommendations.py

def detect_hallucinated_numbers(ai_text: str, rbm: RuleBasedMetrics) -> list[str]:
    """
    Scan NUMBERS: lines for values NOT in Layer 3 metrics.
    If Gemma invents a number, flag it.
    """
    allowed = _build_allowed_numbers(rbm)  # All Layer 3 numbers
    
    suspicious = []
    numbers_lines = re.findall(r"NUMBERS:\s*(.+?)(?:\n|$)", ai_text)
    
    for line in numbers_lines:
        tokens = re.findall(r"\b\d+\.?\d*\b", line)
        for tok in tokens:
            val = float(tok)
            # Check if this number is in allowed set (with 2% tolerance)
            is_close = any(
                abs(val - a) <= max(abs(a) * 0.02, 0.001)
                for a in allowed
            )
            if not is_close:
                suspicious.append(f"{tok} ← HALLUCINATED")
    
    return suspicious


def sanitize_numbers_lines(ai_text: str, rbm: RuleBasedMetrics) -> str:
    """
    Safety net: Replace EVERY NUMBERS: line with correct Python-computed values.
    Even if Gemma gets it wrong, we fix it deterministically.
    """
    # Pre-computed correct values from Layer 3
    correct_numbers = f"{rbm.per_passenger_g_per_km:.1f} g/pax-km → {rbm.train_baseline_g_per_pkm:.1f} g/pax-km → save {rbm.vs_train_saving_pct}%"
    
    # Replace every NUMBERS line
    result = re.sub(
        r"NUMBERS:\s*[^\n]+",
        f"NUMBERS: {correct_numbers}",
        ai_text
    )
    return result
```

**Real Example from Your Run:**
```
Gemma Writes:
  "30.0 g/pax-km → 14.0 g/pax-km → save 53.3%"

Layer 5 Checks:
  ✓ Is 30.0 in allowed list? YES (from Layer 3)
  ✓ Is 14.0 in allowed list? YES (from Layer 3)
  ✓ Is 53.3% in allowed list? YES (from Layer 3)
  
Result: ALL NUMBERS VERIFIED ✓

If Gemma made up:
  "30.0 g/pax-km → 14.0 g/pax-km → save 99%"
  
Layer 5 Detects:
  ✗ Is 99% in allowed list? NO
  
Action: Replace with correct value
  "30.0 g/pax-km → 14.0 g/pax-km → save 53.3%"
```

---

## **COMPLETE DATA FLOW**

```
USER ENTERS 13 PARAMETERS
    ↓
┌─────────────────────────────────────────────────┐
│ LAYER 1: input_validation.py                    │
│ ✓ Check capacity, distance, physics            │
│ ✓ Auto-correct if needed                        │
│ Output: Clean validated data                    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ LAYER 2: training.py + app.py                   │
│ Random Forest Classifier                        │
│ Output: EMISSION LEVEL (LOW/MEDIUM/HIGH)       │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ LAYER 3: emission_analysis.py                   │
│ Calculate: 30.0 g/pax-km, train 14.0, save 53% │
│ Output: 4 PRECISE METRICS (numbers only)        │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ LAYER 4: ollama_recommendations.py              │
│ (Ollama.generate + Gemma 3:1b)                  │
│ Output: 3 HUMAN-READABLE RECOMMENDATIONS        │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ LAYER 5: ollama_recommendations.py              │
│ (detect_hallucinated_numbers +                  │
│  sanitize_numbers_lines)                        │
│ Output: VERIFIED SAFE RECOMMENDATIONS           │
└─────────────────────────────────────────────────┘
    ↓
DISPLAY RESULTS TO USER
```

---

## **Quick Reference Table**

| Layer | File | Function | Input | Output |
|---|---|---|---|---|
| **1** | `input_validation.py` | `validate_inputs()` | 13 raw parameters | Clean data OR error |
| **2** | `training.py` + `app.py` | `model.predict()` | Validated 13 params | LOW/MEDIUM/HIGH |
| **3** | `emission_analysis.py` | `run_rule_based_analysis()` | Classification + metrics | 4 calculated numbers |
| **4** | `ollama_recommendations.py` | `ollama.generate()` | Numbers + location | Gemma's text output |
| **5** | `ollama_recommendations.py` | `detect_hallucinated_numbers()` + `sanitize_numbers_lines()` | Gemma's text | Verified safe text |

---

## **For Your Viva**

You can now say:

> **"My system is organized into 5 separate layers across 3 main files:**
>
> **Layer 1 (input_validation.py):** Police checking if your data makes sense
> **Layer 2 (training.py + app.py):** Brain classifying the emission level
> **Layer 3 (emission_analysis.py):** Engineer calculating exact numbers
> **Layer 4 (ollama_recommendations.py):** Ollama/Gemma explaining in words
> **Layer 5 (ollama_recommendations.py):** Inspector verifying Gemma didn't cheat
>
> **This is why it's a robust Decision Support System, not just an AI chatbot.**"

---

Does this file mapping make it clearer? 🎯
