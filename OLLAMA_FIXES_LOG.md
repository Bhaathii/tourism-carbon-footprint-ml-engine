# 🔧 Ollama Recommendations Logic - Fixes Applied

## Issues Fixed ✅

### Issue 1: "Switch to Train" for Train Users
**Problem:** User on Train was told to "switch to train" (redundant & illogical)  
**Root Cause:** Logic only checked if user was greener than baseline, not if they were already ON an optimal vehicle  
**Solution:** Added `vehicle_is_already_optimal` detection in `RuleBasedMetrics`

### Issue 2: Equivalent Alternatives Treated as Upgrades
**Problem:** CTB Bus & e-tuk (14 g/pax-km) shown as alternatives when Train is 14 g/pax-km  
**Root Cause:** Recommendation logic didn't account for vehicle type matching  
**Solution:** Occupancy sharing prioritized when vehicle is already optimal

### Issue 3: Non-Occupancy Recommendations for Optimal Vehicles
**Problem:** Train users shown "switch to bus" instead of "add passengers"  
**Root Cause:** Prompt hints didn't differentiate between vehicle modes  
**Solution:** Updated `build_dss_prompt()` with three-tier logic

## Code Changes

### 1. `emission_analysis.py` - RuleBasedMetrics
```python
@dataclass
class RuleBasedMetrics:
    # ... existing fields ...
    vehicle_is_already_optimal: bool  # NEW: True if on Train/Bus/e-tuk/Bicycle
```

### 2. `emission_analysis.py` - run_rule_based_analysis()
```python
# Detect if vehicle is already in optimal low-emission mode
OPTIMAL_VEHICLES = {
    "Train",
    "Public Bus (CTB)",
    "Electric Tuk-tuk (e-tuk)",
    "Bicycle",
}
vehicle_is_optimal = vehicle_type in OPTIMAL_VEHICLES
```

### 3. `emission_analysis.py` - build_dss_prompt()
```python
# Three-tier logic:
if rbm.vehicle_is_already_optimal:
    # All 3 recommendations focus on occupancy sharing
    n1, hint1 = _n_occ, "fill all seats on your vehicle"
    n2, hint2 = _n_occ, "book group/shared tickets"
    n3, hint3 = _n_occ, "carpool or join organised tour"
elif rbm.current_is_greener_than_train:
    # Good vehicle but not standard mode — focus on occupancy
    n1, hint1 = _n_occ, "share vehicle / fill all seats"
    n2, hint2 = _n_occ, "use shared tuk-tuk or group transport"
    n3, hint3 = _n_occ, "carpool or join organised tour group"
else:
    # High-emission vehicle — recommend better modes
    n1, hint1 = _n_train, "switch to train or public transport"
    n2, hint2 = _n_occ,   "share vehicle / fill to full occupancy"
    n3, hint3 = _n_train, "use CTB bus or e-tuk instead of car"
```

### 4. `ollama_recommendations.py` - sanitize_numbers_lines()
Updated to use new `vehicle_is_already_optimal` flag for proper number slot assignment

## Test Results ✅

### Test 1: Train User (Already Optimal)
- ✅ `vehicle_is_already_optimal: True`
- ✅ Hints: "fill all seats", "book group/shared tickets", "carpool"
- ✅ NO "switch to train" recommendation
- ✅ Metrics: 27.7 → 13.9 g/pax-km (50% saving by sharing)

### Test 2: Private Car User (Non-Optimal)  
- ✅ `vehicle_is_already_optimal: False`
- ✅ Hints: "switch to train or public transport"
- ✅ Shows 94.9% saving from mode-switching
- ✅ Occupancy saving as secondary option

### Test 3: CTB Bus User (Already Optimal)
- ✅ `vehicle_is_already_optimal: True`
- ✅ All recommendations focus on group booking
- ✅ Occupancy-focused, not mode-switching

## Impact

| Scenario | Before | After | Impact |
|----------|--------|-------|--------|
| Train user (1 pax) | ❌ "Switch to train" | ✅ "Fill all seats" | User is already on best mode |
| Bus user | ❌ "Try e-tuk" | ✅ "Book groups" | Promotes shared transport |
| Car user | ✅ "Try train" | ✅ "Try train" (with 95% saving) | Maintains good suggestions |

## Files Modified
- ✅ `code/emission_analysis.py` (RuleBasedMetrics, run_rule_based_analysis, build_dss_prompt)
- ✅ `code/ollama_recommendations.py` (sanitize_numbers_lines)
- ✅ `code/__init__.py` (created for imports)
