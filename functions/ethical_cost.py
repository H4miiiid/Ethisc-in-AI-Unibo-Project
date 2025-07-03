from typing import Optional, Dict
import math
import logging

# Importing constants for ethical cost adjustments
from .constants import (
    SCHOOL_RUN_MORNING,
    SCHOOL_RUN_AFTERNOON,
    SCHOOL_RUN_LATE,
    FRAGILITY_SENSITIVE_HOURS,
    FRAGILITY_INDEX_MIN,
    FRAGILITY_INDEX_MAX,
    FRAGILITY_CATEGORIES,
    FRAGILITY_WEIGHTS
)

logger = logging.getLogger(__name__)

def ethical_cost_function(
    raw_value: float,
    female_percentage: float,
    fragility_index: float,
    time_of_day: Optional[float] = None,  # 0–24
    params: Optional[Dict] = None

) -> float:
    """Adjust simulator value using ethical considerations.

    The female share and fragility index are smoothed through logistic
    curves so their influence grows gradually instead of linearly. Time
    windows for school runs and fragility sensitivity have weekday and
    weekend variants defined in :mod:`constants`.

    Parameters
    ----------
    raw_value : float
        Original simulator value.
    female_percentage : float
        Share of female population in percent (``0``-``100``).
    fragility_index : float
        Economic fragility score. Typical range is
        ``[FRAGILITY_INDEX_MIN, FRAGILITY_INDEX_MAX]``.
    time_of_day : float, optional
        Hour of day in the ``[0, 24)`` range. ``None`` disables
        time-based adjustments.
    is_weekend : bool, optional
        If ``True``, weekend time windows (``SCHOOL_RUN_WEEKEND`` and
        ``WEEKEND_FRAGILITY_SENSITIVE_HOURS``) are used. Defaults to ``False``.
    params : dict, optional
        Configuration overrides for the adjustment factors.

    Returns
    -------
    float
        Adjusted value after applying ethical coefficients.
    """

    """
    The logistic_smoothing function maps a value in the range [0, 1] to another value in [0, 1] using a logistic (sigmoid-like) curve —
    but with tuned steepness and normalization to make the curve more centered and bounded within [0, 1].
    """

    def logistic_smoothing(x: float, k: float = 6.0) -> float:
        """Map ``x`` in ``[0, 1]`` to ``[0, 1]`` using a logistic curve."""
        x = max(0.0, min(x, 1.0))
        exp_neg = math.exp(-k * (x - 0.5))
        min_v = 1 / (1 + math.exp(k / 2))
        max_v = 1 / (1 + math.exp(-k / 2))
        logistic = 1 / (1 + exp_neg)
        return (logistic - min_v) / (max_v - min_v)

    if params is None:
        params = {}

    logger.debug("Initial value: %f", raw_value)

    if not 0 <= female_percentage <= 100:
        raise ValueError("female_percentage must be between 0 and 100")
    if fragility_index < 0:
        raise ValueError("fragility_index must be >= 0")
    if time_of_day is not None and not (0 <= time_of_day < 24):
        raise ValueError("time_of_day must be in [0, 24)")

    value = raw_value

    # Normalize female percentage
    female_pct = max(0.0, min(female_percentage / 100, 1.0))  # Clamp between 0 and 1
    logger.debug("Normalized female percentage: %.3f", female_pct)

    # --- BASE FEMALE ADJUSTMENT (applies at all times) ---
    female_base_reduction = params.get('female_reduction_base', 0.05)  # Default: 5% max
    female_factor = 1 - female_base_reduction * logistic_smoothing(female_pct)
    value *= female_factor
    logger.debug("After base female adjustment (factor %.3f): %f", female_factor, value)

    # --- SCHOOL RUN FEMALE ADJUSTMENT (time-specific) ---
    if time_of_day is not None:

        school_run_reduction = params.get('school_run_female_reduction', 0.1)  # 10% max
        
        morning_school_run = params.get('school_run_morning', SCHOOL_RUN_MORNING)
        afternoon_pickup = params.get('school_run_afternoon', SCHOOL_RUN_AFTERNOON)
        late_pickup = params.get('school_run_late', SCHOOL_RUN_LATE)
        
        in_window = (
            morning_school_run[0] <= time_of_day < morning_school_run[1]
            or afternoon_pickup[0] <= time_of_day < afternoon_pickup[1]
            or late_pickup[0] <= time_of_day < late_pickup[1]
        )

        if in_window:
            school_run_factor = 1 - school_run_reduction * logistic_smoothing(female_pct)
            value *= school_run_factor
            logger.debug(
                "Applied school run adjustment (factor %.3f) at %.2f h: %f",
                school_run_factor,
                time_of_day,
                value,
            )

    # --- FRAGILITY ADJUSTMENT ---
    fragility_base = params.get('fragility_reduction_base', 0.2)
    fragility_min = params.get('fragility_index_min', FRAGILITY_INDEX_MIN)
    fragility_max = params.get('fragility_index_max', FRAGILITY_INDEX_MAX)
    fragility_span = max(1e-6, fragility_max - fragility_min)

    # Clamping the fragility index
    fragility_ratio = max(
        0.0,
        min((fragility_index - fragility_min) / fragility_span, 1.0),
    )
    logger.debug(
        "Fragility ratio after clamping: %.3f (index %.2f)",
        fragility_ratio,
        fragility_index,
    )

    ratio_effect = logistic_smoothing(fragility_ratio)
    logger.debug("Fragility logistic effect: %.3f", ratio_effect)

    if time_of_day is not None:
        
        off_peak = params.get('fragility_sensitive_hours', FRAGILITY_SENSITIVE_HOURS)
        early_or_late = time_of_day < 7 or time_of_day >= 20

        if off_peak[0] <= time_of_day < off_peak[1]:
            fragility_factor = 1 - (fragility_base * 1.2) * ratio_effect
            logger.debug(
                "Off-peak fragility adjustment (factor %.3f) at %.2f h",
                fragility_factor,
                time_of_day,
            )
        elif early_or_late:
            fragility_factor = 1 - (fragility_base * 0.7) * ratio_effect
            logger.debug(
                "Early/late fragility adjustment (factor %.3f) at %.2f h",
                fragility_factor,
                time_of_day,
            )
        else:
            fragility_factor = 1 - fragility_base * ratio_effect
            logger.debug(
                "Daytime fragility adjustment (factor %.3f) at %.2f h",
                fragility_factor,
                time_of_day,
            )
    else:
        fragility_factor = 1 - fragility_base * ratio_effect
        logger.debug("Fragility adjustment without time (factor %.3f)", fragility_factor)

    value *= fragility_factor
    logger.debug("Final value after fragility adjustment: %f", value)

    # --- ADJUST FOR MOBILITY (smoothed sigmoid scaling) ---
    if params.get("adjust_for_mobility", False):  # or pass this as an argument if preferred
        vehicle_penalty_factor = params.get("fragility_mobility_penalty", 0.4)
        mobility_scaling = 1 - vehicle_penalty_factor * math.tanh(2 * fragility_ratio)
        
        logger.debug(
            "Mobility penalty adjustment: tanh(2 * %.2f) = %.3f → scaling factor: %.3f",
            fragility_ratio,
            math.tanh(2 * fragility_ratio),
            mobility_scaling
        )

        value *= mobility_scaling
        logger.debug("Value after mobility adjustment: %f", value)

# Return final adjusted value
    return value


def classify_fragility_emission_weight(fragility_score):
    """
    Returns a weight multiplier for emissions based on economic fragility.
    """
    thresholds = FRAGILITY_CATEGORIES
    if thresholds[0] <= fragility_score < thresholds[1]:
        return FRAGILITY_WEIGHTS["LOW"]
    elif thresholds[1] <= fragility_score < thresholds[2]:
        return FRAGILITY_WEIGHTS["Medium-Low"]
    elif thresholds[2] <= fragility_score < thresholds[3]:
        return FRAGILITY_WEIGHTS["Medium"]
    elif thresholds[3] <= fragility_score < thresholds[4]:
        return FRAGILITY_WEIGHTS["Medium-High"]
    elif thresholds[4] <= fragility_score <= thresholds[5]:
        return FRAGILITY_WEIGHTS["High"] 
    else:
        return FRAGILITY_WEIGHTS["Fallback-Default"]  # Default weight for out-of-range scores
