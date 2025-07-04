
CRS_LATLONG = "EPSG:4326"
CRS_PROJECTED = "EPSG:6875"
# P2V = 0.48994377844432835
P2V = 0.6265460762750291

# ---------------------------------------------------------------------------
#  Time windows and fragility thresholds used by :func:`ethical_cost_function`.
#  Defined here so they can be tuned centrally.
# ---------------------------------------------------------------------------

# School run windows on weekdays (morning drop-off, afternoon pick-up, late pick-up)
SCHOOL_RUN_MORNING = (7.0, 9.0)
SCHOOL_RUN_AFTERNOON = (13.0, 15.0)
SCHOOL_RUN_LATE = (16.0, 18.0)

# Hours when fragile populations are most active
FRAGILITY_SENSITIVE_HOURS = (10.0, 15.0)

# Expected fragility index range and categories
FRAGILITY_INDEX_MIN = 81.0
FRAGILITY_INDEX_MAX = 120.0

# Thresholds dividing fragility into categories (Low .. High)
FRAGILITY_CATEGORIES = [81.0, 89.0, 97.0, 105.0, 113.0, 120.0]

FRAGILITY_INDEX_MIN = FRAGILITY_CATEGORIES[0]
FRAGILITY_INDEX_MAX = FRAGILITY_CATEGORIES[-1]

# Fragility emission weights for each category
FRAGILITY_WEIGHTS = {
    "Low": 1.00,
    "Medium-Low": 1.10,
    "Medium": 1.20,
    "Medium-High": 1.30,
    "High": 1.40,
    "Fallback-Default": 1.00
}
