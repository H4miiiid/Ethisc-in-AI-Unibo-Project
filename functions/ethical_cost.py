from typing import Optional, Dict

def ethical_cost_function(
    raw_value: float,
    female_percentage: float,
    fragility_index: float,
    time_of_day: Optional[float] = None,  # 0–24
    params: Optional[Dict] = None
) -> float:
    """
    Adjust simulator value using ethical considerations: gender, fragility, and time.
    Applies a base adjustment for female population and fragility,
    and extra adjustment for time-sensitive contexts (e.g., school runs, off-peak).
    """

    if params is None:
        params = {}

    value = raw_value

    # Normalize female percentage
    female_pct = max(0.0, min(female_percentage / 100, 1.0))  # Clamp between 0 and 1

    # --- BASE FEMALE ADJUSTMENT (applies at all times) ---
    female_base_reduction = params.get('female_reduction_base', 0.05)  # Default: 5% max
    female_factor = 1 - (female_base_reduction * female_pct)
    value *= female_factor

    # --- SCHOOL RUN FEMALE ADJUSTMENT (time-specific) ---
    if time_of_day is not None:
        morning_school_run = params.get('school_run_morning', (7, 9))
        afternoon_pickup = params.get('school_run_afternoon', (13, 15))
        late_pickup = params.get('school_run_late', (16, 18))
        school_run_reduction = params.get('school_run_female_reduction', 0.1)  # 10% max

        if (morning_school_run[0] <= time_of_day < morning_school_run[1] or
            afternoon_pickup[0] <= time_of_day < afternoon_pickup[1] or
            late_pickup[0] <= time_of_day < late_pickup[1]):
            school_run_factor = 1 - (school_run_reduction * female_pct)
            value *= school_run_factor

    # --- FRAGILITY ADJUSTMENT ---
    fragility_base = params.get('fragility_reduction_base', 0.2)
    fragility_max = max(1e-6, params.get('fragility_index_max', 1.0))  # Prevent divide by zero

    # Clamp fragility index
    fragility_ratio = max(0.0, min(fragility_index / fragility_max, 1.0))

    if time_of_day is not None:
        off_peak = params.get('fragility_sensitive_hours', (10, 15))
        early_or_late = time_of_day < 7 or time_of_day >= 20

        if off_peak[0] <= time_of_day < off_peak[1]:
            fragility_factor = 1 - (fragility_base * fragility_ratio * 1.2)
        elif early_or_late:
            fragility_factor = 1 - (fragility_base * fragility_ratio * 0.7)
        else:
            fragility_factor = 1 - (fragility_base * fragility_ratio)
    else:
        fragility_factor = 1 - (fragility_base * fragility_ratio)

    value *= fragility_factor

    return value
