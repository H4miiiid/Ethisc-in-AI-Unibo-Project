import itertools
import math
import numbers
import statistics
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import geopandas as gpd
from itertools import combinations

from dt_model import Index, ContextVariable, UniformDistIndex
from sympy import Piecewise, Function, lambdify, exp

### VARIABLES
DECISION_STRATEGY = "parallel" ## choices are taken sequentially or parallely? use values in ['sequential', 'parallel']
TIME_SHIFT_STRATEGY = "flexible" ## choice: flexibly shift their activity time (earlier or later), or fixed and predetermined time shift. use values in ['fixed', 'flexible']
MODAL_SHIFT_OPTION = "tpm" ## choice: if the modal shift is included, and if yes which type. use values in ['no', 'tpm', 'active', 'tpm-active']         
TRAFFIC_COMPUTATION_MODE = "deterministic" ## choice of the method to compute traffic. use values in ['adaptive', 'deterministic']

# Variable check
if DECISION_STRATEGY not in ['sequential', 'parallel']:
    raise ValueError(f"DECISION_STRATEGY is '{DECISION_STRATEGY}', but should have values in ['parallel','sequential']")
if TIME_SHIFT_STRATEGY not in ['fixed', 'flexible']:
    raise ValueError(f"TIME_SHIFT_STRATEGY is '{TIME_SHIFT_STRATEGY}', but should have values in ['fixed', 'flexible']")
if MODAL_SHIFT_OPTION not in ['no', 'tpm', 'active', 'tpm-active']:
    raise ValueError(f"MODAL_SHIFT_OPTION is '{MODAL_SHIFT_OPTION}', but should have values in ['no', 'tpm', 'active', 'tpm-active']")
if TRAFFIC_COMPUTATION_MODE not in ['deterministic', 'adaptive']:
    raise ValueError(f"TRAFFIC_COMPUTATION_MODE is '{TRAFFIC_COMPUTATION_MODE}', but should have values in ['adaptive', 'deterministic']")


### LOAD THE INPUT DATA ###

# Load the shape of the zones and the AV # TODO: from data lake
gdf_area_verde = gpd.read_file("area_verde_manual_v1.geojson")
aree_gdf_inside = gpd.read_file("aree_gdf_inside_v2.geojson")
aree_gdf_outside = gpd.read_file("aree_gdf_outside_v2.geojson")
aree_gdf = pd.concat([aree_gdf_inside, aree_gdf_outside], axis=0)
aree_gdf_AV = gpd.overlay(aree_gdf, gdf_area_verde, how='intersection')
aree_statistiche_gdf = gpd.read_file("aree-statistiche.geojson") # Load the aree_statistiche.geojson file

# FIXME: fix missing name of some zones
if aree_gdf_AV[aree_gdf_AV['id'] == 48]['name'].iloc[0] is None:
    aree_gdf_AV.loc[aree_gdf_AV['id'] == 48, 'name'] = aree_gdf_AV.loc[aree_gdf_AV['id'] == 48, 'code']
else:
    raise ValueError("Area Verde shape fix not applicable")

# Load the zone names and codes
zones = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52,
         53, 54, 55, 60, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87,
         89, 90, 91, 92, 93, 94, 95, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
         115, 116, 117, 118, 120, 121, 123, 124, 125, 126, 127, 128, 129, 10116, 10131, 10137]

zones_to_names = {id: aree_gdf_AV[aree_gdf_AV['id'] == id]['name'].iloc[0] for id in zones}
names_to_zones = {aree_gdf_AV[aree_gdf_AV['id'] == id]['name'].iloc[0]: id for id in zones}

# Load the inflow and starting estimate # TODO: from data lake
vehicle_inflow = np.array(
    [221.25, 208.70090169, 196.59507377, 184.93251626,
     173.71322915, 162.93721244, 152.60446612, 142.71499021,
     133.2687847, 124.26584959, 115.70618488, 107.58979057,
     99.91666667, 92.68681316, 85.90023005, 79.55691735,
     73.65687504, 68.20010313, 63.18660163, 58.57940003,
     54.34152786, 50.4729851, 46.97377176, 43.84388784,
     41.08333333, 38.69210825, 36.67021258, 35.01764633,
     33.73440951, 32.82050209, 32.2759241, 31.97296144,
     31.78390001, 31.70873982, 31.74748086, 31.90012315,
     32.16666667, 32.54711142, 33.04145742, 33.64970465,
     34.37185312, 35.20790282, 36.15785376, 37.26801653,
     38.58470172, 40.10790933, 41.83763935, 43.7738918,
     45.91666667, 48.26596395, 50.82178366, 53.58412578,
     56.55299032, 59.72837728, 63.11028667, 67.02310605,
     71.79122301, 77.41463755, 83.89334968, 91.22735938,
     99.41666667, 108.46127153, 118.36117398, 129.11637401,
     140.72687162, 153.19266681, 166.51375958, 181.49843978,
     198.95499726, 218.88343202, 241.28374407, 266.15593339,
     293.5, 323.31594389, 355.60376506, 390.36346351,
     427.59503924, 467.29849225, 509.47382255, 552.31727381,
     594.02508973, 634.59727031, 674.03381555, 712.33472545,
     749.5, 785.52963921, 820.42364308, 854.18201161,
     886.80474479, 918.29184263, 948.64330513, 976.28773216,
     999.65372359, 1018.74127943, 1033.55039966, 1044.0810843,
     1050.33333333, 1052.30714677, 1050.00252461, 1043.41946686,
     1032.5579735, 1017.41804455, 997.99967999, 977.02114803,
     957.20071686, 938.53838647, 921.03415686, 904.68802804,
     889.5, 875.47007275, 862.59824628, 850.88452059,
     840.32889569, 830.93137158, 822.69194825, 815.37241668,
     808.73456786, 802.77840178, 797.50391844, 792.91111785,
     789., 785.77056489, 783.22281253, 781.35674292,
     780.17235604, 779.66965191, 779.84863052, 780.46888892,
     781.29002415, 782.3120362, 783.53492508, 784.9586908,
     786.58333333, 788.4088527, 790.43524889, 792.66252192,
     795.09067177, 797.71969844, 800.54960195, 803.02721276,
     804.59936133, 805.26604768, 805.02727179, 803.88303368,
     801.83333333, 798.87817076, 795.01754596, 790.25145892,
     784.57990966, 778.00289817, 770.52042444, 763.0159827,
     756.37306713, 750.59167774, 745.67181454, 741.61347751,
     738.41666667, 736.081382, 734.60762352, 733.99539122,
     734.24468509, 735.35550515, 737.32785139, 739.61300218,
     741.6622359, 743.47555253, 745.0529521, 746.39443459,
     747.5, 748.36964834, 749.0033796, 749.40119378,
     749.56309089, 749.48907093, 749.17913389, 749.28980052,
     750.47759157, 752.74250705, 756.08454694, 760.50371126,
     766., 772.57341316, 780.22395074, 788.95161274,
     798.75639917, 809.63831001, 821.59734528, 834.56717618,
     848.48147392, 863.34023851, 879.14346994, 895.89116821,
     913.58333333, 932.2199653, 951.8010641, 972.32662975,
     993.79666225, 1016.21116159, 1039.57012777, 1062.26316093,
     1082.67986119, 1100.82022857, 1116.68426305, 1130.27196464,
     1141.58333333, 1150.61836914, 1157.37707205, 1161.85944207,
     1164.06547919, 1163.99518343, 1161.64855477, 1157.67561751,
     1152.72639595, 1146.80089008, 1139.89909992, 1132.02102544,
     1123.16666667, 1113.33602359, 1102.5290962, 1090.74588451,
     1077.98638852, 1064.25060823, 1049.53854363, 1034.03498586,
     1017.92472607, 1001.20776426, 983.88410042, 965.95373456,
     947.41666667, 928.27289675, 908.52242481, 888.16525085,
     867.20137486, 845.63079684, 823.4535168, 800.82280064,
     777.89191427, 754.66085769, 731.12963089, 707.29823388,
     683.16666667, 658.73492924, 634.0030216, 608.97094374,
     583.63869568, 558.0062774, 532.07368891, 506.86154363,
     483.39045498, 461.66042296, 441.67144757, 423.4235288,
     406.91666667, 392.15086116, 379.12611228, 367.84242003,
     358.29978441, 350.49820541, 344.43768305, 339.53803014,
     335.21905953, 331.48077121, 328.32316518, 325.74624144,
     323.75, 322.33444085, 321.49956399, 321.24536942,
     321.57185714, 322.47902716, 323.96687946, 325.32925698,
     325.86000264, 325.55911644, 324.42659838, 322.46244845,
     319.66666667, 316.03925302, 311.58020751, 306.28953013,
     300.1672209, 293.2132798, 285.42770685, 276.81050203,
     267.36166534, 257.0811968, 245.9690964, 234.02536413])

vehicle_starting = np.array(
    [469.1171328, 443.69548489, 419.09959956, 395.3285459,
     372.38106805, 350.25383391, 328.94375292, 308.44819234,
     288.76128005, 269.8762396, 251.78508616, 234.47653096,
     217.93708628, 202.14904537, 187.23364639, 173.33538936,
     160.74133748, 149.35005648, 138.9484643, 129.82635441,
     121.86598415, 114.77473984, 108.30635234, 102.36660628,
     97.16858696, 92.57561054, 88.42882506, 84.9448748,
     82.11069425, 79.58333193, 77.5308616, 75.98326301,
     74.80147557, 74.06382282, 73.62570768, 73.39617194,
     73.20922493, 73.35049, 73.77324004, 74.70100345,
     76.11377869, 78.1950941, 80.86988539, 84.1994823,
     88.34546937, 93.11287441, 98.37821869, 104.4567326,
     111.30556965, 118.90798256, 127.38109469, 136.71400294,
     147.19850848, 158.96430308, 172.74985618, 188.67018516,
     206.53799311, 226.19393513, 247.48651269, 270.21151144,
     294.82326303, 320.9848637, 348.15693932, 377.26242984,
     408.67499197, 443.6879152, 482.90415449, 526.40939645,
     575.4673876, 630.5359741, 691.99486404, 759.30690206,
     831.84679481, 909.53516619, 992.46510547, 1081.25324145,
     1173.23263127, 1267.31224022, 1364.385995, 1462.72929862,
     1562.97276692, 1663.41149118, 1761.77046943, 1856.24064997,
     1944.70763995, 2025.39263178, 2098.21347757, 2164.03157446,
     2222.80669116, 2273.48438739, 2315.08911371, 2346.38202128,
     2368.67843287, 2382.08418138, 2385.99931324, 2380.52794821,
     2366.94203708, 2345.79971447, 2318.22555807, 2284.72580097,
     2245.20222551, 2203.06226955, 2160.58956689, 2118.46569656,
     2079.22750844, 2041.41607667, 2005.03172182, 1971.0502901,
     1939.648819, 1912.60281951, 1890.38957103, 1873.10769447,
     1859.78148484, 1848.09949687, 1838.82995101, 1832.17746538,
     1828.16038744, 1826.51808786, 1826.21735968, 1826.88007429,
     1828.39874915, 1829.97937969, 1831.9997007, 1833.99369168,
     1835.93182816, 1837.87919191, 1839.59113463, 1841.96620398,
     1844.7362667, 1847.28073761, 1849.24014321, 1851.09931602,
     1853.19725001, 1855.19382063, 1856.53563205, 1856.05883598,
     1854.94215668, 1852.53783865, 1850.37603606, 1847.62877204,
     1843.30421379, 1837.17846012, 1829.8732786, 1822.0078322,
     1813.28749445, 1804.67059695, 1795.46792375, 1785.45796328,
     1775.88200036, 1767.33748236, 1759.81284511, 1754.31933693,
     1749.66948686, 1745.61678166, 1743.22800768, 1742.59776868,
     1743.94918182, 1745.7116133, 1749.05968815, 1752.41008736,
     1756.83030659, 1760.49176821, 1763.90883261, 1766.99951358,
     1769.39584602, 1770.97625561, 1773.56852645, 1776.38038594,
     1778.85975329, 1782.2013791, 1785.37939945, 1789.42351973,
     1794.34570127, 1800.64273285, 1808.5826721, 1819.02606153,
     1831.28830973, 1845.65899969, 1863.24776831, 1883.72164382,
     1907.44118734, 1933.05725526, 1960.80428054, 1987.41740859,
     2014.4712567, 2042.39166814, 2072.5155976, 2106.33447882,
     2143.0250319, 2180.90762014, 2219.30359959, 2257.55233811,
     2295.50058022, 2333.36668476, 2368.58085581, 2399.09334357,
     2424.29216816, 2445.45334539, 2462.6604712, 2475.8334986,
     2484.59287628, 2487.94201616, 2486.32796115, 2479.98421509,
     2468.80993952, 2453.25160052, 2433.6751749, 2410.11160915,
     2385.05647488, 2360.40997268, 2337.06571142, 2315.82406424,
     2295.76955681, 2276.06490635, 2257.9385966, 2241.28583533,
     2226.28249517, 2212.01558544, 2196.48011568, 2179.73254121,
     2162.87568558, 2146.31768362, 2129.42113472, 2111.16633946,
     2090.90368935, 2067.58560761, 2041.11327124, 2012.64983009,
     1981.63352362, 1947.75371966, 1910.31014311, 1868.9113914,
     1823.77345504, 1776.3958802, 1726.31116668, 1674.19311091,
     1620.77185539, 1565.16114384, 1507.89083843, 1448.97780561,
     1390.15982787, 1331.60992067, 1273.98666373, 1217.4900917,
     1162.83665329, 1110.51401921, 1061.01560972, 1015.06383647,
     972.2161952, 932.84820362, 897.53349296, 867.24168728,
     840.53357721, 817.12167681, 796.76957585, 779.17538322,
     764.10398194, 751.85304426, 741.9860572, 734.07151811,
     728.20817781, 723.34565396, 719.66960917, 717.20425799,
     715.89884563, 715.49402594, 715.64805251, 716.40935477,
     717.33242379, 718.40851324, 719.31175849, 719.26786956,
     717.7663086, 715.35734916, 711.54710803, 705.86680218,
     697.68578456, 688.20500293, 677.37969251, 665.15237417,
     651.44865567, 636.17305282, 619.20255347, 600.38112308,
     579.50875306, 556.3263212, 530.50311514, 501.61506706])

# Load the inflow, starting and traffic for each zone # TODO: from data lake
zone_io = pd.read_parquet('AreaVerde_IO_v13.parquet')
# zone_starting = pd.read_parquet('AreaVerde_S_v13.parquet')
zone_traffic = pd.read_parquet('AreaVerde_T_v13.parquet')

# Load the proportions of euro class
euro_class_split = {
    'euro_0': 0.059,
    'euro_1': 0.012,
    'euro_2': 0.034,
    'euro_3': 0.054,
    'euro_4': 0.198,
    'euro_5': 0.176,
    'euro_6': 0.467
}

# Load the emissions (nox) per car euro level per km
euro_class_emission = {
    'euro_0': 0.210584391986267347,
    'euro_1': 0.2174573179869368,
    'euro_2': 0.24014520073869067,
    'euro_3': 0.24723923486567853,
    'euro_4': 0.1355550834386541,
    'euro_5': 0.09955851060544411,
    'euro_6': 0.06824599009858062
}

# parameter for modeling and approximation
P_PROB_THRESHOLD = 0.005  # In all cdf, values below the threshold are set to 0
P_TOTAL_TRAFFIC = 2200368.245435709  # Sum of traffic
P_RECORD_FREQUENCY = 12  # Number of measures per hour [12 times]
P_RECORD_HEADWAY = 1/P_RECORD_FREQUENCY  # Delta time between measures, expressed in hour [5 min]
P_DWELL_TIME_AV = 1/3  # Average dwell time in AV, expressed in hours [20 min]
P_PT_comfort = 0.5 # 1 - Average load factor on the PTS as a measure of passengers' comfort
P_PT_FREQUENCY = 0.3259 # Average frequency of the PT vehicles as a measure of PTS' availability
P_PT_CAPILLARITY = 0.1962 # Capillarity of the PTS over the road network as a measure of PTS availability
P_PT_AVG_COST = 3.0743 # Average fee of using the PT as a measure of the PTS' cost


### AREA VERDE MODEL ###

### Define base functions (used in the model) ###

class TS_sum(Function):
    pass


def ts_sum(ts):
    return np.expand_dims(ts.sum(axis=1), axis=1)


class TS_solve(Function):
    pass


def ts_solve_deterministic(
    ts, 
    dwell_time=P_DWELL_TIME_AV
):
    tau = dwell_time * P_RECORD_FREQUENCY     

    decay = np.exp(-np.arange(ts.shape[1], 0, -1) / tau)
    decay[decay <= P_PROB_THRESHOLD] = 0

    series = np.zeros_like(ts)
    for t in range(ts.shape[1]):
        decay_shifted = np.roll(decay, t)
        series[:, t] = ts[:, t] + ts_sum(ts * decay_shifted)[:, 0]
    
    return series


def ts_solve_adaptive(
    ts, dwell_time=P_DWELL_TIME_AV, max_iter=50, min_diff=1e-5
):
    tau = dwell_time * P_RECORD_FREQUENCY

    series = ts.copy()
    for iter in range(max_iter):
        mu = 1 + (tau-1)
        alfa = (mu - 1) / mu

        next_series = ts + np.roll(series, 1, axis=1) * alfa

        if np.max(np.abs(series - next_series)) < min_diff:
            break

        series = next_series
    
    return series


ts_solve = {
    'adaptive': ts_solve_adaptive,
    'deterministic': ts_solve_deterministic
}


class TS_b_choose(Function):
    pass


def ts_b_choose(w_a, *list_w_b):

    if len(list_w_b) == 0:
        return np.ones_like(w_a)

    # Initialize the result array with zeros
    p = np.zeros_like(w_a)
    
    # Process all combinations of w_b being active, from 0 to len(list_w_b)
    for r in range(len(list_w_b) + 1):
        for indices in combinations(range(len(list_w_b)), r):
            p_tmp = w_a.copy()
            
            for i, w_b in enumerate(list_w_b):
                if i in indices:
                    p_tmp = p_tmp * w_b  # This w_b is active
                else:
                    p_tmp = p_tmp * (1-w_b)  # This w_b is inactive
            
            if r > 0:
                denominator = w_a.copy()
                for idx in indices:
                    denominator = denominator + list_w_b[idx]
                # Avoid division by zero
                weight = np.divide(w_a, denominator, out=np.zeros_like(denominator), where=denominator != 0)
                p_tmp = p_tmp * weight

            p = p + p_tmp
    
    return p


class TS_anticipate(Function):
    pass

def ts_anticipate(number_anticipating, delta_from_start, p50_anticipating):
    
    # Identify the policy starting time
    t0 = np.where(delta_from_start[0,:] == 0)[0][0]
    tmax = np.shape(number_anticipating)[1]

    # Define the exponential decay
    range1 = np.arange(start=0, stop=number_anticipating.shape[1], step=1) * P_RECORD_HEADWAY
    range2 = np.arange(1, number_anticipating.shape[1]+1) * P_RECORD_HEADWAY
    v1 = (np.exp(range1 / p50_anticipating * np.log(0.5)) - 
          np.exp(range2 / p50_anticipating * np.log(0.5)))
    v1 = np.where(v1 < P_PROB_THRESHOLD, 0, v1)

    number_anticipated = np.zeros_like(number_anticipating)
    for t in range(t0, tmax):
        # Restrict the v1
        v1_here = v1[:, (t-t0):]
        v1_here_sum = ts_sum(v1_here)
        zero_indices = np.where(v1_here_sum[:, 0] != 0)[0]
        if len(zero_indices) > 0:
            v1_normalized = np.where(v1_here_sum > 0, v1_here / v1_here_sum, 0)
            
            # Update the anticipated adding the ones from time t to time t0-deltat
            for deltat in range(1,t0+1):
                weight = v1_normalized[:, deltat-1]
                number_anticipated[:, t0-deltat] = number_anticipated[:, t0-deltat] + number_anticipating[:, t] * weight

    return number_anticipated


class TS_postpone(Function):
    pass

def ts_postpone(number_postponing, delta_to_end, p50_postponing):

    # Identify the policy starting time
    t0 = np.where(delta_to_end[0,:] == 0)[0][0]
    tmax = np.shape(number_postponing)[1]
    
    # Define the exponential decay
    range1 = np.arange(start=0, stop=number_postponing.shape[1], step=1) * P_RECORD_HEADWAY
    range2 = np.arange(start=1, stop=number_postponing.shape[1]+1, step=1) * P_RECORD_HEADWAY
    v1 = (np.exp(range1 / p50_postponing * np.log(0.5)) - 
          np.exp(range2 / p50_postponing * np.log(0.5)))
    v1 = np.where(v1 < P_PROB_THRESHOLD, 0, v1)

    number_postponed = np.zeros_like(number_postponing)
    for t in range(t0, -1, -1):
        number_postponing_here = number_postponing[:, t]

        # Restrict the v1
        v1_here = v1[:, (t0-t):]
        v1_here_sum = ts_sum(v1_here)
        zero_indices = np.where(v1_here_sum[:, 0] != 0)[0]
        if len(zero_indices) > 0:
            v1_normalized = np.where(v1_here_sum > 0, v1_here / v1_here_sum, 0)
            
            for deltat in range(tmax-t0-1):
                weight = v1_normalized[:, deltat]
                number_postponed[:, t0+1+deltat] = number_postponed[:, t0+1+deltat] + number_postponing_here * weight

    return number_postponed


def upsample(values):
    df = pd.DataFrame(values, columns=['values'])
    df['n'] = df.index * 12 + 6
    df = df.set_index('n').reindex(range(24 * 12))
    df.loc[0, 'values'] = (values[0] + values[-1]) / 2
    df.loc[24 * 12 - 1, 'values'] = df.loc[0, 'values']
    # TODO: min 0
    df = df.interpolate(method='cubic')
    df[df['values'] < 0] = 0
    return df['values'].values


### Model definition ###

# TS specific functin map (form Function symbol to implementation)
TS_map = {'TS_sum': ts_sum,
          'TS_solve': ts_solve[TRAFFIC_COMPUTATION_MODE],
          'TS_b_choose': ts_b_choose,
          'TS_anticipate': ts_anticipate,
          'TS_postpone': ts_postpone,
          }

# "macro" to extend index to time series
def TS_Index(name: str, value: Any, cvs: list[ContextVariable] | None = None) -> Index:
    index = Index(name, value, cvs=cvs)
    index.value = lambdify(cvs, value, ["numpy", TS_map ])
    return index

TS = Index('time range', 
           np.array([(t - pd.Timestamp('00:00:00')).total_seconds() 
                     for t in pd.date_range(start='00:00:00', periods=12 * 24, freq='5min')]))

class Model:
    def __init__(self):
        self.TS = TS

        # Parameters
        self.I_P_start_time = Index('start time', (pd.Timestamp('07:30:00') - pd.Timestamp('00:00:00')).total_seconds())
        self.I_P_end_time = Index('end time', (pd.Timestamp('19:30:00') - pd.Timestamp('00:00:00')).total_seconds())

        self.I_P_cost = [Index(f'cost euro {e}', 5.00 - e * 0.25) for e in range(7)]
        self.I_P_fraction_exempted = Index('exempted vehicles %', 0.15)

        self.I_B_p50_cost = UniformDistIndex('cost 50% threshold', loc=4.00, scale=7.00)
        self.I_B_p50_anticipating = Index('anticipation 50% likelihood', 0.25)
        self.I_B_p50_postponing = Index('postponement 50% likelihood', 0.5)
        self.I_B_p50_anticipation = Index('anticipation distribution 50% threshold', 0.50)
        self.I_B_p50_postponement = Index('postponement distribution 50% threshold', 0.50)
        self.I_B_starting_modified_factor = Index('starting modified factor', 0.00)

        self.I_B_pt_comfort = Index('importance level of the comfort of the pt', 1.00)
        self.I_B_pt_capillarity = Index('importance level of capillarity of the pt', 1.00)
        self.I_B_pt_frequency = Index('importance level of frequency per stop of the pt', 1.00)
        self.I_B_pt_cost = Index('importance level of the cost of the pt', 1.00)

        # Presence / flow variables
        self.TS_inflow = Index('inflow', vehicle_inflow)
        self.TS_starting = Index('starting', vehicle_starting)

        self.TS_inflow_zone_from_inside = {}
        self.TS_inflow_zone_from_outside = {}
        self.TS_inflow_zone = {}
        self.TS_traffic_zone = {}

        for zone in zones:
            zone_inside_inflow_values = zone_io[zone_io['id_zone'] == zone]['inflow_from_INSIDE_mean'].values / 12
            zone_outside_inflow_values = zone_io[zone_io['id_zone'] == zone]['inflow_from_OUTSIDE_mean'].values / 12
            zone_traffic_values = zone_traffic[zone_traffic['id_zone'] == zone]['traffic_in_zone_mean'].values / 12

            self.TS_inflow_zone_from_inside[zone] = Index(f'zone {zone} inflow from inside', upsample(zone_inside_inflow_values))
            self.TS_inflow_zone_from_outside[zone] = Index(f'zone {zone} inflow from outside', upsample(zone_outside_inflow_values))
            self.TS_inflow_zone[zone] = Index(f'zone {zone}  inflow', self.TS_inflow_zone_from_inside[zone] +
                                         self.TS_inflow_zone_from_outside[zone],
                                         [self.TS_inflow_zone_from_inside[zone], self.TS_inflow_zone_from_outside[zone]])
            self.TS_traffic_zone[zone] = Index(f'zone {zone}  traffic', upsample(zone_traffic_values))

        # Indices - current state
        self.I_traffic = TS_Index('reference traffic',
                                  TS_solve(self.TS_inflow + self.TS_starting),
                                  cvs=[self.TS_inflow, self.TS_starting])

        self.I_total_base_inflow = TS_Index('total base vehicle inflow',
                                            TS_sum(self.TS_inflow),
                                            cvs=[self.TS_inflow])

        self.I_average_emissions = Index('average emissions (per vehicle, per km)',
                            euro_class_emission['euro_0'] * euro_class_split['euro_0'] +
                            euro_class_emission['euro_1'] * euro_class_split['euro_1'] +
                            euro_class_emission['euro_2'] * euro_class_split['euro_2'] +
                            euro_class_emission['euro_3'] * euro_class_split['euro_3'] +
                            euro_class_emission['euro_4'] * euro_class_split['euro_4'] +
                            euro_class_emission['euro_5'] * euro_class_split['euro_5'] +
                            euro_class_emission['euro_6'] * euro_class_split['euro_6'])

        # Indices - time
        self.I_delta_from_start = Index('delta time from start',
                                   Piecewise(((self.TS - self.I_P_start_time) / pd.Timedelta('1h').total_seconds(), self.TS >= self.I_P_start_time),
                                             (np.inf, True)),
                                   cvs=[self.TS, self.I_P_start_time])
        
        self.I_delta_to_end = Index('delta time to end',
                               Piecewise(((self.I_P_end_time - self.TS) / pd.Timedelta('1h').total_seconds(), self.TS <= self.I_P_end_time),
                                         (np.inf, True)),
                               cvs=[self.TS, self.I_P_end_time])

        self.I_delta_before_start = Index('delta time before start',
                                     Piecewise(
                                         ((self.I_P_start_time - self.TS) / pd.Timedelta('1h').total_seconds(), self.TS < self.I_P_start_time),
                                         (np.inf, True)),
                                     cvs=[self.TS, self.I_P_start_time])

        self.I_delta_after_end = Index('delta time after end',
                                  Piecewise(((self.TS - self.I_P_end_time) / pd.Timedelta('1h').total_seconds(), self.TS > self.I_P_end_time),
                                            (np.inf, True)),
                                  cvs=[self.TS, self.I_P_end_time])

        # Indices - modified state
        self.I_pt_cost = Index('probability of accepting the cost of the pt',
                                exp(P_PT_AVG_COST / self.I_B_p50_cost * np.log(0.5)),
                                cvs=[self.I_B_p50_cost])
        
        if DECISION_STRATEGY == "parallel":
            self.I_fraction_p_rigid_euro_class = [Index(f'fraction of possibly rigid vehicles per euro_{e} %',
                                                exp(self.I_P_cost[e] / self.I_B_p50_cost * np.log(0.5)),
                                                cvs=[self.I_P_cost[e], self.I_B_p50_cost]) 
                                        for e in range(7)]

            self.I_fraction_p_rigid = Index('fraction of possibly rigid vehicles',
                                    (self.I_fraction_p_rigid_euro_class[0] * euro_class_split['euro_0'] +
                                    self.I_fraction_p_rigid_euro_class[1] * euro_class_split['euro_1'] +
                                    self.I_fraction_p_rigid_euro_class[2] * euro_class_split['euro_2'] +
                                    self.I_fraction_p_rigid_euro_class[3] * euro_class_split['euro_3'] +
                                    self.I_fraction_p_rigid_euro_class[4] * euro_class_split['euro_4'] +
                                    self.I_fraction_p_rigid_euro_class[5] * euro_class_split['euro_5'] +
                                    self.I_fraction_p_rigid_euro_class[6] * euro_class_split['euro_6']) *
                                    (1 - self.I_P_fraction_exempted),
                                    cvs=[self.I_P_fraction_exempted, *self.I_fraction_p_rigid_euro_class])

            self.I_fraction_p_anticipating = Index('fraction of possibly anticipating vehicles',
                                            exp(self.I_delta_from_start / self.I_B_p50_anticipating * np.log(0.5)) *
                                            (1 - self.I_P_fraction_exempted),
                                            cvs=[self.I_delta_from_start, self.I_B_p50_anticipating, self.I_P_fraction_exempted])

            self.I_fraction_p_postponing = Index('fraction of possibly postponing vehicles',
                                        exp(self.I_delta_to_end / self.I_B_p50_postponing * np.log(0.5)) *
                                        (1 - self.I_P_fraction_exempted),
                                        cvs=[self.I_delta_to_end, self.I_B_p50_postponing, self.I_P_fraction_exempted])
            
            if MODAL_SHIFT_OPTION == "tpm" or MODAL_SHIFT_OPTION == "tpm+active":
                self.I_fraction_p_mode_shifted = Index('fraction of possibly mode-shifted vehicles',
                                                        1 / (1 + exp(self.I_B_pt_comfort * P_PT_comfort + self.I_B_pt_capillarity * P_PT_CAPILLARITY +
                                                            self.I_B_pt_frequency * P_PT_FREQUENCY + self.I_B_pt_cost * self.I_pt_cost)) * 
                                                        (1 - self.I_P_fraction_exempted),
                                                        cvs=[self.I_B_pt_comfort, self.I_B_pt_capillarity, self.I_B_pt_frequency, self.I_B_pt_cost, self.I_pt_cost, self.I_P_fraction_exempted])
            elif MODAL_SHIFT_OPTION == "no" or MODAL_SHIFT_OPTION == "active":
                self.I_fraction_p_mode_shifted = Index('fraction of possibly mode-shifted vehicles', 0.0)
                
            self.I_fraction_rigid = TS_Index('fraction of rigid vehicles',
                                        Piecewise(
                                            (TS_b_choose(self.I_fraction_p_rigid, self.I_fraction_p_anticipating, self.I_fraction_p_postponing, self.I_fraction_p_mode_shifted)*(1 - self.I_P_fraction_exempted),
                                                (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                            (1 - self.I_P_fraction_exempted, 
                                                True)),
                                        cvs=[self.I_fraction_p_rigid, self.I_fraction_p_anticipating, self.I_fraction_p_postponing, self.I_fraction_p_mode_shifted, self.I_P_fraction_exempted, self.TS, self.I_P_start_time, self.I_P_end_time])

            self.I_fraction_rigid_euro_class = [Index(f'fraction rigid vehicles per euro_{e} %',
                                                    self.I_fraction_rigid * 
                                                        (euro_class_split[f'euro_{e}'] * self.I_fraction_p_rigid_euro_class[e]) /
                                                        (self.I_fraction_p_rigid / (1-self.I_P_fraction_exempted)) ,
                                                    cvs=[self.I_fraction_rigid, self.I_fraction_p_rigid_euro_class[e],
                                                        self.I_fraction_p_rigid, self.I_P_fraction_exempted]) 
                                                for e in range(7)]
            
            self.I_fraction_anticipating = TS_Index('fraction of anticipating vehicles',
                                                Piecewise(
                                                (TS_b_choose(self.I_fraction_p_anticipating, self.I_fraction_p_rigid, self.I_fraction_p_postponing, self.I_fraction_p_mode_shifted)*
                                                    (1-self.I_P_fraction_exempted),
                                                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                                (0,
                                                        True)),
                                                cvs=[self.I_fraction_p_anticipating, self.I_fraction_p_rigid, self.I_fraction_p_postponing, self.I_fraction_p_mode_shifted, self.I_P_fraction_exempted, self.TS, self.I_P_start_time, self.I_P_end_time])

            self.I_fraction_postponing = TS_Index('fraction of postponing vehicles',
                                                Piecewise(
                                                    (TS_b_choose(self.I_fraction_p_postponing, self.I_fraction_p_rigid, self.I_fraction_p_anticipating, self.I_fraction_p_mode_shifted)*
                                                    (1-self.I_P_fraction_exempted),
                                                        (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                                    (0,
                                                        True)),
                                                cvs=[self.I_fraction_p_postponing, self.I_fraction_p_rigid, self.I_fraction_p_anticipating, self.I_fraction_p_mode_shifted, self.I_P_fraction_exempted, self.TS, self.I_P_start_time, self.I_P_end_time])

            self.I_fraction_mode_shifted = TS_Index('fraction of mode-shifted vehicles',
                                                Piecewise(
                                                    (TS_b_choose(self.I_fraction_p_mode_shifted, self.I_fraction_p_postponing, self.I_fraction_p_rigid, self.I_fraction_p_anticipating)*(1-self.I_P_fraction_exempted),
                                                        (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                                    (0,
                                                        True)), 
                                                cvs=[self.I_fraction_p_mode_shifted, self.I_fraction_p_postponing, self.I_fraction_p_rigid, self.I_fraction_p_anticipating, self.I_P_fraction_exempted, self.TS, self.I_P_start_time, self.I_P_end_time])

            self.I_fraction_lost = Index('fraction of lost vehicles',
                                    Piecewise(
                                        ((1-self.I_fraction_p_rigid)*(1-self.I_fraction_p_anticipating)*(1-self.I_fraction_p_postponing)*(1-self.I_fraction_p_mode_shifted) *
                                              (1 - self.I_P_fraction_exempted),
                                              (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                        (0,
                                            True)),
                                     cvs=[self.I_fraction_p_rigid, self.I_fraction_p_anticipating, self.I_fraction_p_postponing, self.I_fraction_p_mode_shifted, self.I_P_fraction_exempted, self.TS, self.I_P_start_time, self.I_P_end_time])

            
        elif DECISION_STRATEGY == "sequential":
            self.I_fraction_rigid_euro_class = [Index(f'rigid vehicles euro_{e} %',
                                                    (1 - self.I_P_fraction_exempted) * exp(self.I_P_cost[e] / self.I_B_p50_cost * np.log(0.5))* euro_class_split[f'euro_{e}'],
                                                    cvs=[self.I_P_fraction_exempted, self.I_P_cost[e], self.I_B_p50_cost]) 
                                                for e in range(7)]
            
            self.I_fraction_rigid = Index('rigid vehicles %',
                                            self.I_fraction_rigid_euro_class[0] +
                                            self.I_fraction_rigid_euro_class[1] +
                                            self.I_fraction_rigid_euro_class[2] +
                                            self.I_fraction_rigid_euro_class[3] +
                                            self.I_fraction_rigid_euro_class[4] +
                                            self.I_fraction_rigid_euro_class[5] +
                                            self.I_fraction_rigid_euro_class[6] ,
                                            cvs=[*self.I_fraction_rigid_euro_class])            
            
            self.I_fraction_anticipating = Index('anticipating vehicles %',
                                                 Piecewise(
                                                    (exp(self.I_delta_from_start / self.I_B_p50_anticipating * np.log(0.5)) *
                                                        (1 - self.I_P_fraction_exempted - self.I_fraction_rigid),
                                                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                                    (0, True)),
                                                cvs=[self.I_delta_from_start, self.I_B_p50_anticipating, self.I_P_fraction_exempted, self.I_fraction_rigid,
                                                     self.TS, self.I_P_start_time, self.I_P_end_time])
            
            self.I_fraction_postponing = Index('postponing vehicles %',
                                                Piecewise(
                                                    (exp(self.I_delta_to_end / self.I_B_p50_postponing * np.log(0.5)) *
                                                        (1 - self.I_P_fraction_exempted - self.I_fraction_rigid - self.I_fraction_anticipating),
                                                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                                    (0, True)),
                                                cvs=[self.I_delta_to_end, self.I_B_p50_postponing, self.I_P_fraction_exempted, self.I_fraction_rigid, self.I_fraction_anticipating,
                                                     self.TS, self.I_P_start_time, self.I_P_end_time])
            
            if MODAL_SHIFT_OPTION == "tpm" or MODAL_SHIFT_OPTION == "tpm+active":
                self.I_fraction_mode_shifted = Index('fraction of mode-shifted vehicles',
                                                Piecewise(
                                                    (1 / (1 + exp(self.I_B_pt_comfort * P_PT_comfort + self.I_B_pt_capillarity * P_PT_CAPILLARITY +
                                                            self.I_B_pt_frequency * P_PT_FREQUENCY + self.I_B_pt_cost * self.I_pt_cost)) *
                                                            (1 - self.I_P_fraction_exempted - self.I_fraction_rigid - self.I_fraction_anticipating - self.I_fraction_postponing),
                                                        (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                                    (0, True)), 
                                                     cvs=[self.I_B_pt_comfort, self.I_B_pt_capillarity, self.I_B_pt_frequency, self.I_B_pt_cost, self.I_pt_cost,
                                                          self.I_P_fraction_exempted, self.I_fraction_rigid, self.I_fraction_anticipating, self.I_fraction_postponing,
                                                          self.TS, self.I_P_start_time, self.I_P_end_time])
            else:
                self.I_fraction_mode_shifted = Index('fraction of mode-shifted vehicles', 0)

            self.I_fraction_lost = Index('fraction of lost vehicles',
                                    Piecewise(
                                        (1 -self.I_fraction_rigid -self.I_fraction_anticipating -self.I_fraction_postponing -self.I_P_fraction_exempted - self.I_fraction_mode_shifted,
                                              (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                        (0,
                                            True)),
                                     cvs=[self.I_fraction_rigid, self.I_fraction_anticipating, self.I_fraction_postponing, self.I_fraction_mode_shifted, self.I_P_fraction_exempted, 
                                          self.TS, self.I_P_start_time, self.I_P_end_time])
           

        self.I_number_anticipating = Index('anticipating vehicles', self.I_fraction_anticipating * self.TS_inflow,
                                      cvs=[self.I_fraction_anticipating, self.TS_inflow])
        
        self.I_total_anticipating = TS_Index('total anticipating vehicles', TS_sum(self.I_number_anticipating), cvs=[self.I_number_anticipating])

        if TIME_SHIFT_STRATEGY == "flexible":
            # TODO: manage (1.0 / 12) constant
            self.I_number_anticipated = (
                Index('anticipated vehicles',
                    ((exp(-(self.I_delta_before_start - (1.0 / 12)) / self.I_B_p50_anticipation * np.log(2)) -
                        exp(-self.I_delta_before_start / self.I_B_p50_anticipation * np.log(2))) *
                    self.I_total_anticipating),
                    cvs=[self.I_delta_before_start, self.I_B_p50_anticipation, self.I_total_anticipating])
            )

            self.I_total_anticipated = TS_Index('total vehicles anticipated', TS_sum(self.I_number_anticipated), cvs=[self.I_number_anticipated])

        elif TIME_SHIFT_STRATEGY == "fixed":
            self.I_number_anticipated = TS_Index('vehicles anticipated',
                                          TS_anticipate(self.I_number_anticipating, self.I_delta_from_start, self.I_B_p50_anticipating), #TODO: check number_anticipating!
                                          cvs=[self.I_number_anticipating, self.I_delta_from_start, self.I_B_p50_anticipating])

            self.I_total_anticipated = TS_Index('total vehicles anticipated',
                                            TS_sum(self.I_number_anticipated), 
                                            cvs=[self.I_number_anticipated])

        self.I_number_postponing = Index('postponing vehicles', self.I_fraction_postponing * self.TS_inflow,
                                    cvs=[self.I_fraction_postponing, self.TS_inflow])

        self.I_total_postponing = TS_Index('total postponing vehicles', TS_sum(self.I_number_postponing), cvs=[self.I_number_postponing])

        if TIME_SHIFT_STRATEGY == "flexible":
            # TODO: manage (1.0 / 12) constant
            # TODO: postponing and postponed still do not coincide since we truncate at midnight
            self.I_number_postponed = (
                Index('postponed vehicles',
                    ((exp(-(self.I_delta_after_end - (1.0 / 12)) / self.I_B_p50_postponement * np.log(2)) -
                        exp(-self.I_delta_after_end / self.I_B_p50_postponement * np.log(2))) *
                    self.I_total_postponing),
                    cvs=[self.I_delta_after_end, self.I_B_p50_postponement, self.I_total_postponing])
            )

            self.I_total_postponed = TS_Index('total vehicles postponed', TS_sum(self.I_number_postponed), cvs=[self.I_number_postponed])

        elif TIME_SHIFT_STRATEGY == "fixed":
            self.I_number_postponed = TS_Index('vehicles postponed',
                                        TS_postpone(self.I_number_postponing, self.I_delta_to_end, self.I_B_p50_postponing),#TODO: check number_anticipating!
                                        cvs=[self.I_number_postponing, self.I_delta_to_end, self.I_B_p50_postponing])

            self.I_total_postponed = TS_Index('total vehicles postponed',
                                        TS_sum(self.I_number_postponed), 
                                        cvs=[self.I_number_postponed])

        self.I_number_time_shifted = Index('time-shifted vehicles', 
                                           self.I_number_anticipated + self.I_number_postponed,
                                           cvs=[self.I_number_anticipated, self.I_number_postponed])
        
        self.I_total_time_shifted = Index('total vehicles shifted',
                                     self.I_total_anticipated + self.I_total_postponed,
                                     cvs=[self.I_total_anticipated, self.I_total_postponed])
        
        self.I_number_mode_shifted = Index('mode-shifted vehicles', 
                                           self.I_fraction_mode_shifted * self.TS_inflow,
                                      cvs=[self.I_fraction_mode_shifted, self.TS_inflow])
    
        self.I_total_mode_shifted = TS_Index('total mode shifted vehicles', TS_sum(self.I_number_mode_shifted), cvs=[self.I_number_mode_shifted])

        self.I_number_lost = Index('lost vehicles', 
                                    self.I_fraction_lost * self.TS_inflow,
                                    cvs=[self.I_fraction_lost, self.TS_inflow])
    
        self.I_total_lost = TS_Index('total lost vehicles', TS_sum(self.I_number_lost), cvs=[self.I_number_lost])

        self.I_modified_inflow = Index('modified vehicle inflow',
                               Piecewise(((self.I_P_fraction_exempted + self.I_fraction_rigid) * self.TS_inflow,
                                          (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                         (self.TS_inflow + self.I_number_time_shifted, True)),
                               cvs=[self.I_P_fraction_exempted, self.I_fraction_rigid, self.I_P_start_time, self.I_P_end_time, self.TS, self.TS_inflow,
                                    self.I_number_time_shifted])

        self.I_total_modified_inflow = TS_Index('total modified vehicle inflow', TS_sum(self.I_modified_inflow), cvs=[self.I_modified_inflow])

        self.I_modified_starting = Index('modified starting', 
                                         self.TS_starting + self.TS_starting * self.I_B_starting_modified_factor * 
                                            ((self.I_total_modified_inflow / self.I_total_base_inflow) - 1) ,
                                         cvs=[self.TS_starting, self.I_total_modified_inflow, self.I_total_base_inflow, self.I_B_starting_modified_factor])

        # TODO: flat rates for all AV
        self.I_inflow_ratio = Index('ratio between modified flow and base flow',
                                    self.TS_inflow / self.I_modified_inflow, 
                                    cvs=[self.TS_inflow, self.I_modified_inflow])

        self.I_starting_ratio = Index('ratio between modified starting and base starting',
                                      self.TS_starting / self.I_modified_starting, 
                                      cvs=[self.TS_starting, self.I_modified_starting])

        self.I_modified_inflow_zone = {zone: Index(f'modified zone {zone} inflow',
                                           self.TS_inflow_zone_from_outside[zone] / self.I_inflow_ratio +
                                           self.TS_inflow_zone_from_inside[zone] / self.I_starting_ratio,
                                           [self.TS_inflow_zone_from_outside[zone], self.I_inflow_ratio,
                                            self.TS_inflow_zone_from_inside[zone], self.I_starting_ratio])
                               for zone in zones}

        self.I_delta_inflow_zone = {zone: Index(f'delta zone {zone} inflow',
                                         self.I_modified_inflow_zone[zone] - self.TS_inflow_zone[zone],
                                         cvs=[self.I_modified_inflow_zone[zone], self.TS_inflow_zone[zone]])
                             for zone in zones}

        self.I_modified_traffic = TS_Index('modified traffic',
                                        TS_solve(self.I_modified_inflow + self.I_modified_starting),
                                        cvs=[self.I_modified_inflow, self.I_modified_starting])

        self.I_traffic_ratio = Index('ratio between modified traffic and base traffic',
                                self.I_traffic / self.I_modified_traffic, 
                                cvs=[self.I_traffic, self.I_modified_traffic])

        # TODO: TS_traffic_zone is an input, while I_traffic is calculated
        self.I_modified_traffic_zone = {zone: Index(f'modified zone {zone} traffic', self.TS_traffic_zone[zone] / self.I_traffic_ratio,
                                              [self.TS_traffic_zone[zone], self.I_traffic_ratio])
                                  for zone in zones}

        self.I_delta_traffic_zone = {zone: Index(f'delta zone {zone} traffic',
                                            self.I_modified_traffic_zone[zone] - self.TS_traffic_zone[zone],
                                            [self.I_modified_traffic_zone[zone] , self.TS_traffic_zone[zone]])
                                  for zone in zones}

        self.I_number_paying = Index('paying vehicles',
                                     Piecewise(
                                        (self.I_fraction_rigid * self.TS_inflow, (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                        (0, True)),
                                     cvs=[self.I_fraction_rigid, self.I_P_start_time, self.I_P_end_time, self.TS, self.TS_inflow])

        self.I_total_paying = TS_Index('total vehicles paying', TS_sum(self.I_number_paying), cvs=[self.I_number_paying])

        self.I_modified_euro_class_split = [Index(f'modified split euro_{e} %',
                                             Piecewise(
                                                (((self.I_P_fraction_exempted*euro_class_split[f'euro_{e}']) + self.I_fraction_rigid_euro_class[e]) / 
                                                (self.I_P_fraction_exempted + self.I_fraction_rigid),
                                                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time)),
                                                (euro_class_split[f'euro_{e}'], 
                                                    True)
                                             ),
                                             cvs=[self.I_P_fraction_exempted, self.I_fraction_rigid_euro_class[e], self.I_fraction_rigid, 
                                                  self.TS, self.I_P_start_time, self.I_P_end_time]) 
                                            for e in range(7)]

        self.I_modified_avg_cost_per_payers = Index('modified average cost with respect to the vehicles paying',
                                        self.I_P_cost[0] * self.I_fraction_rigid_euro_class[0] +
                                        self.I_P_cost[1] * self.I_fraction_rigid_euro_class[1] +
                                        self.I_P_cost[2] * self.I_fraction_rigid_euro_class[2] +
                                        self.I_P_cost[3] * self.I_fraction_rigid_euro_class[3] +
                                        self.I_P_cost[4] * self.I_fraction_rigid_euro_class[4] +
                                        self.I_P_cost[5] * self.I_fraction_rigid_euro_class[5] +
                                        self.I_P_cost[6] * self.I_fraction_rigid_euro_class[6],
                                        cvs=[*self.I_P_cost, *self.I_fraction_rigid_euro_class])
         
        self.I_total_paid = Index('total paid fees', 
                                   self.I_total_paying * self.I_modified_avg_cost_per_payers,
                                   cvs=[self.I_total_paying, self.I_modified_avg_cost_per_payers])

        self.I_modified_average_emissions = Index('modified average emissions (per vehicle, per km)',
                                            euro_class_emission['euro_0'] * self.I_modified_euro_class_split[0] +
                                            euro_class_emission['euro_1'] * self.I_modified_euro_class_split[1] +
                                            euro_class_emission['euro_2'] * self.I_modified_euro_class_split[2] +
                                            euro_class_emission['euro_3'] * self.I_modified_euro_class_split[3] +
                                            euro_class_emission['euro_4'] * self.I_modified_euro_class_split[4] +
                                            euro_class_emission['euro_5'] * self.I_modified_euro_class_split[5] +
                                            euro_class_emission['euro_6'] * self.I_modified_euro_class_split[6],
                                            cvs=[*self.I_modified_euro_class_split])
      
        # TODO: improve - at the moment, the conversion factor is 2,5 km per 5 minutes
        self.I_emissions = Index('emissions', 
                                 2.5 * self.I_average_emissions * self.I_traffic, 
                                 cvs=[self.I_average_emissions, self.I_traffic])

        # I_modified_emissions = Index('modified emissions', 2.5 * I_modified_average_emissions * I_modified_traffic,
        #                             cvs=[I_modified_average_emissions, I_modified_traffic])
        #
        # TODO: The average emissions is probably different outside regulated hours
        #  (shifted cars' emissions are probably proportional to shifted cars' euro level mix)
        #
        self.I_modified_emissions = Index('modified emissions',
                                          2.5 * self.I_modified_average_emissions * self.I_modified_traffic,
                                          cvs=[self.I_modified_average_emissions, 
                                              self.I_modified_traffic, self.I_P_start_time,self.I_P_end_time, self.TS])

        self.I_emissions_zone = {zone: Index(f'zone {zone} emissions', 2.5 * self.I_average_emissions * self.TS_traffic_zone[zone],
                                        cvs=[self.I_average_emissions, self.TS_traffic_zone[zone]])
                            for zone in zones}

        self.I_modified_emissions_zone = {zone: Index(f'modified zone {zone} emissions',
                                                2.5 * self.I_modified_average_emissions * self.I_modified_traffic_zone[zone],
                                                cvs=[self.I_modified_average_emissions, self.I_modified_traffic_zone[zone]])
                                    for zone in zones}
        self.I_delta_emissions_zone = {zone: Index(f'delta zone {zone} emissions',
                                              self.I_modified_emissions_zone[zone] - self.I_emissions_zone[zone],
                                              cvs=[self.I_modified_emissions_zone[zone], self.I_emissions_zone[zone]])
                                  for zone in zones}

        self.I_total_emissions = TS_Index('total emissions',
                                       TS_sum(self.I_emissions), 
                                       cvs=[self.I_emissions])

        self.I_total_modified_emissions = TS_Index('total modified emissions',
                                                TS_sum(self.I_modified_emissions),
                                                cvs=[self.I_modified_emissions])

        self.indices_parameters = [
            self.I_P_start_time, self.I_P_end_time, *self.I_P_cost, self.I_P_fraction_exempted, self.I_B_p50_cost,
            self.I_B_p50_anticipating, self.I_B_p50_anticipation, self.I_B_p50_postponing, self.I_B_p50_postponement,
            self.I_B_starting_modified_factor, self.I_B_pt_comfort, self.I_B_pt_capillarity, self.I_B_pt_frequency,
            self.I_B_pt_cost
        ]

        self.indices_current_totals = [
            self.TS_inflow, self.TS_starting, *self.TS_inflow_zone_from_inside.values(),
            *self.TS_inflow_zone_from_outside.values(), *self.TS_inflow_zone.values(), *self.TS_traffic_zone.values(),
            self.I_traffic, self.I_total_base_inflow, self.I_average_emissions
        ]

        self.indices_time = [
            self.I_delta_from_start, self.I_delta_before_start, self.I_delta_to_end, self.I_delta_after_end
        ]

        if DECISION_STRATEGY == "parallel":
            self.indices_fractions = [
                *self.I_fraction_p_rigid_euro_class, self.I_fraction_p_rigid, self.I_fraction_p_anticipating,
                self.I_fraction_p_postponing, self.I_pt_cost, self.I_fraction_p_mode_shifted,
                self.I_fraction_rigid, *self.I_fraction_rigid_euro_class, self.I_fraction_anticipating,
                self.I_fraction_postponing, self.I_fraction_mode_shifted, self.I_fraction_lost,
            ]
        elif DECISION_STRATEGY == "sequential":
            self.indices_fractions = [
                self.I_pt_cost, *self.I_fraction_rigid_euro_class, self.I_fraction_rigid, self.I_fraction_anticipating,
                self.I_fraction_postponing, self.I_fraction_mode_shifted, self.I_fraction_lost,
            ]

        self.indices_modified_totals = [
            self.I_number_anticipating, self.I_total_anticipating, self.I_number_postponing, self.I_total_postponing,
            self.I_number_anticipated, self.I_number_postponed, self.I_total_anticipated, self.I_total_postponed,
            self.I_number_time_shifted, self.I_total_time_shifted, self.I_number_mode_shifted, self.I_total_mode_shifted,
            self.I_number_lost, self.I_total_lost, self.I_modified_inflow, self.I_total_modified_inflow,
            self.I_modified_starting, self.I_inflow_ratio, self.I_starting_ratio, *self.I_modified_inflow_zone.values(),
            *self.I_delta_inflow_zone.values(), self.I_modified_traffic, self.I_traffic_ratio, *self.I_modified_traffic_zone.values(),
            *self.I_delta_traffic_zone.values()
        ]
        
        self.indices_costs = [
            self.I_number_paying, self.I_total_paying, *self.I_modified_euro_class_split,
            self.I_modified_avg_cost_per_payers, self.I_total_paid
        ]
            
        self.indices_emissions = [ 
            self.I_modified_average_emissions, self.I_emissions, self.I_modified_emissions, *self.I_emissions_zone.values(),
            *self.I_modified_emissions_zone.values(), *self.I_delta_emissions_zone.values(),
            self.I_total_emissions,self.I_total_modified_emissions
        ]

        self.indexes = [self.TS] + self.indices_parameters + self.indices_current_totals + self.indices_time + self.indices_fractions + self.indices_modified_totals + self.indices_costs + self.indices_emissions

    def evaluate(self, size=1):
        subs = {}
        for index in self.indexes:
            if index.cvs is None:
                if isinstance(index.value, numbers.Number):
                    subs[index] = np.expand_dims(np.array([index.value] * size), axis=1)
                elif isinstance(index.value, np.ndarray):
                    subs[index] = np.expand_dims(index.value, axis=0)
                else:
                    subs[index] = np.expand_dims(np.array(index.value.rvs(size=size)), axis=1)
            else:
                args = [subs[cv] for cv in index.cvs]
                subs[index] = index.value(*args)
        return subs


def distribution(field, size=10000, num=100):
    xx, yy = np.meshgrid(np.linspace(0, size, num + 1), range(field.shape[1]))
    zz = stats.poisson(mu=np.expand_dims(field, axis=2)).cdf(np.expand_dims(xx, axis=0))
    return zz.mean(axis=0)


def to_time(seconds):
    return pd.Timestamp('00:00:00').to_pydatetime() + pd.Timedelta(seconds=seconds)


def to_number(time):
    return (time - pd.Timestamp('00:00:00')).total_seconds()

field_color = (165/256,15/256,21/256)
#field_color = (103/256,0,13/256)
#field_color = (239/256,59/256,44/256)
delta=0.5
field_light_color = ((field_color[0]+delta)/(1+delta),
                     (field_color[1]+delta)/(1+delta),
                     (field_color[2]+delta)/(1+delta))

field_colormap = LinearSegmentedColormap.from_list(
    'mid_red_bar',
    colors=['white',field_light_color,field_color,field_light_color,'white'],
    N=100)

def plot_field_graph(field, horizontal_label, vertical_label, vertical_size=None, vertical_formatter=None,
                     reference_line=None):
    if vertical_size is None:
        vertical_size = roundup(np.max(field))
    dist = distribution(field, vertical_size, 100)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    pcm = ax.pcolormesh(pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
                        np.linspace(0, vertical_size, 100 + 1), dist.T,
                        cmap=field_colormap, vmin=0.0, vmax=1.0)
    if reference_line is not None:
        ax.plot(pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
                reference_line, '--', linewidth=1, color='black', label='Riferimento')
    ax.plot(pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
            field.mean(axis=0), linewidth=1, color='black', label='Modificato (mediana)')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_ticks([0.00, 0.25, 0.50, 0.75, 1.00])
    cbar.set_ticklabels([f"{x}%" for x in [0, 25, 50, 75, 100]])
    ax.set_ylim([0, vertical_size])
    if vertical_formatter is not None:
        ax.yaxis.set_major_formatter(vertical_formatter)
    ax.set_ylabel(vertical_label)
    fig.tight_layout()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    ax.set_xlabel(horizontal_label)
    ax.legend(loc='upper right')
    return fig

def plot_multifield_graph(evals, index, zones, horizontal_label,
                          vertical_label, vertical_size=None, vertical_formatter=None,
                          reference_index=None):
    field = sum([evals[index[z]] for z in zones])
    if reference_index is not None:
        reference_line = sum([evals[reference_index[z]][0] for z in zones])
    else:
        reference_line = None
    return plot_field_graph(field, horizontal_label, vertical_label, vertical_size, vertical_formatter, reference_line)

def plot_map_graph(evals, index, label, time=None, range=None, function=sum):
    if time is None:
        df_zones = pd.DataFrame.from_dict(
            {k: function(evals[v].mean(axis=0)) for k, v in index.items()},
            orient='index')
    else:
        if type(time) is tuple:
            (start,end) = time
        else:
            (start,end) = (time, time+pd.Timedelta(seconds=15*60))
        df_zones = pd.DataFrame.from_dict(
            {k: function(itertools.compress(evals[v].mean(axis=0),
                                            (to_number(start) <= TS.value) &
                                            (TS.value < to_number(end)))) for k, v in index.items()},
            orient='index')

    df_zones.index.name = 'id_zone'
    df_zones.rename(columns={0: 'value'}, inplace=True)
    df_zones = df_zones.merge(aree_gdf_AV, left_on='id_zone', right_on='id', how='left')
    df_zones.set_index("id", inplace=True)
    gdf_zones = gpd.GeoDataFrame(df_zones, geometry='geometry')
    gdf_zones.set_crs(epsg=4326, inplace=True, allow_override=True)
    # TODO: Manage range
    fig = px.choropleth_map(
        gdf_zones, geojson=gdf_zones.geometry, locations=gdf_zones.index, color="value",
        hover_name="name", hover_data={"value": ":d"}, labels={'id': 'Codice zona', 'value': 'Valore'},
        center={"lat": 44.492, "lon": 11.341}, map_style="carto-positron", zoom=11.5,
        opacity=0.5, color_continuous_scale='Reds', # range_color=range,
    )
    fig.update_layout(
        autosize=False, width=800, height=600,
        coloraxis_colorbar_title_text=None,
    )
    return fig

from typing import Literal

def plot_statistical_area_map(
    subs, index, time, function=sum, range_val=None, label="",
    visualization_mode: Literal["raw", "fragility_weighted", "gender_weighted", "ethics_weighted"] = "raw"
):
    """
    Plot a choropleth map of statistical areas based on simulation data,
    optionally weighted by ethical indicators (gender, fragility).

    Parameters:
        subs (pd.DataFrame): Simulation results.
        index (dict): Mapping from zone IDs to row indices in `subs`.
        time (tuple or datetime.time): Time interval (start, end) or specific time.
        function (callable): Aggregation function for values over time.
        range_val (tuple): Min/max values for color normalization.
        label (str): Label for the map (unused).
        visualization_mode (str): How to weight the values.
    
    Returns:
        fig (plotly.Figure): Choropleth map.
    """

    # --- 1. Aggregate simulation values per area ---
    metric_name = list(index.values())[0].name
    function_to_use = sum if 'delta' in metric_name and 'traffic' not in metric_name else function

    if isinstance(time, tuple):  # time is a (start, end) range
        start, end = time

        def agg_zone(z):
            values = subs[index[z]].mean(axis=0)
            mask = (to_number(start) <= TS.value) & (TS.value < to_number(end))
            selected = list(itertools.compress(values, mask))
            return function_to_use(selected) if selected else 0

        area_values = {z: agg_zone(z) for z in index.keys()}
    else:  # time is a single time instance
        time_idx = time.hour * 4 + time.minute // 15
        area_values = {z: subs[index[z]][:, time_idx].sum() for z in index.keys()}

    df = pd.DataFrame.from_dict(area_values, orient='index', columns=['value'])
    df['id'] = list(index.keys())

    # --- 2. Load statistical areas and harmonize CRS ---
    aree_statistiche_gdf = gpd.read_file("aree-statistiche.geojson")

    if aree_gdf_AV.crs != aree_statistiche_gdf.crs:
        aree_statistiche_gdf = aree_statistiche_gdf.to_crs(aree_gdf_AV.crs)

    # Drop columns with nested dictionaries to avoid merge issues
    dict_cols = [
        col for col in aree_statistiche_gdf.columns
        if aree_statistiche_gdf[col].apply(lambda x: isinstance(x, dict)).any()
    ]
    aree_statistiche_gdf = aree_statistiche_gdf.drop(columns=dict_cols)

    # --- 3. Load gender and fragility indicators ---
    gender = pd.read_parquet('gender.parquet')
    fragility = pd.read_parquet('fragilita-2021.parquet')

    total_residents = gender.groupby('area_statistica')['residenti'].sum()
    female_residents = gender[gender['sesso'] == 'Femmine'].groupby('area_statistica')['residenti'].sum()
    female_percentage = (female_residents / total_residents * 100).rename('female_percentage')
    fragility_index = fragility.groupby('area_statistica')['frag_compl'].mean().rename('fragility_index')

    ethics_df = pd.concat([female_percentage, fragility_index], axis=1).reset_index()

    # --- 4. Match simulation zones to statistical areas via spatial intersection ---
    simulation_gdf = aree_gdf_AV.set_index('id').join(df.set_index('id')).reset_index()

    # Use projected CRS for accurate area calculation (e.g., UTM zone for Bologna)
    projected_crs = 'EPSG:32632'
    simulation_gdf_proj = simulation_gdf.to_crs(projected_crs)
    aree_statistiche_gdf_proj = aree_statistiche_gdf.to_crs(projected_crs)

    # Compute intersection areas
    overlaps = gpd.overlay(simulation_gdf_proj, aree_statistiche_gdf_proj, how='intersection')
    overlaps['overlap_area'] = overlaps.geometry.area

    # For each simulation zone, keep only the largest overlapping statistical area
    overlaps = overlaps.sort_values('overlap_area', ascending=False).drop_duplicates(subset='id', keep='first')

    # Aggregate simulation values by statistical area
    df_agg = overlaps.groupby("area_statistica").agg({'value': 'sum'}).reset_index()

    # Merge simulation values back into original statistical area GeoDataFrame
    df_plot = aree_statistiche_gdf.merge(df_agg, on="area_statistica", how="left")
    df_plot.set_index("area_statistica", inplace=True)
    gdf_areas = gpd.GeoDataFrame(df_plot, geometry='geometry')
    gdf_areas.set_crs(epsg=4326, inplace=True, allow_override=True)
    gdf_areas['area_statistica'] = gdf_areas.index
    gdf_areas = gdf_areas.reset_index(drop=True)

    # --- 5. Merge with ethical indicators and compute weights ---
    gdf_areas['area_statistica_norm'] = gdf_areas['area_statistica'].str.strip().str.upper()
    ethics_df['area_statistica_norm'] = ethics_df['area_statistica'].str.strip().str.upper()
    gdf_areas = gdf_areas.merge(ethics_df, on='area_statistica_norm', how='left')

    # Normalize and handle missing data
    gdf_areas['female_norm'] = gdf_areas['female_percentage'] / 100
    gdf_areas['female_norm'] = gdf_areas['female_norm'].fillna(0)

    gdf_areas['fragility_norm'] = gdf_areas['fragility_index'] / gdf_areas['fragility_index'].max()
    gdf_areas['fragility_norm'] = gdf_areas['fragility_norm'].fillna(0)

    # Composite ethics index
    ethics_weights = {'female_norm': 0.5, 'fragility_norm': 0.5}
    gdf_areas['ethics_index'] = sum(
        gdf_areas[indicator] * weight for indicator, weight in ethics_weights.items()
    )
    gdf_areas['ethics_index'] /= sum(ethics_weights.values())  # Normalize to [0, 1]

    # Set final color values based on visualization mode
    gdf_areas['color_value'] = gdf_areas['value']
    if visualization_mode == 'fragility_weighted':
        gdf_areas['color_value'] *= gdf_areas['fragility_norm']
    elif visualization_mode == 'gender_weighted':
        gdf_areas['color_value'] *= gdf_areas['female_norm']
    elif visualization_mode == 'ethics_weighted':
        gdf_areas['color_value'] *= gdf_areas['ethics_index']

    # --- 6. Handle color range filtering and NaNs ---
    range_color = None
    if range_val:
        min_r, max_r = range_val
        min_r = min_r if min_r is not None else gdf_areas['color_value'].min()
        max_r = max_r if max_r is not None else gdf_areas['color_value'].max()
        gdf_areas = gdf_areas[gdf_areas['color_value'].between(min_r, max_r)]
        range_color = [min_r, max_r]

    # Round and sanitize values
    for col in ["value", "color_value"]:
        gdf_areas[col] = gdf_areas[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    gdf_areas["value_rounded"] = gdf_areas["value"].round().astype(int)
    gdf_areas["color_value_rounded"] = gdf_areas["color_value"].round().astype(int)

    # --- 7. Create choropleth plot ---
    fig = px.choropleth_map(
        gdf_areas,
        geojson=gdf_areas.geometry,
        locations=gdf_areas.index,
        color=gdf_areas["color_value"],
        hover_name="area_statistica_norm",
        hover_data={
            "value_rounded": True,
            "color_value_rounded": True,
            "female_percentage": ':.2f',
            "fragility_index": ':.2f',
        },
        labels={
            'area_statistica_norm': 'Area Statistica',
            'value_rounded': 'Valore',
            'color_value_rounded': 'Nuovo_Valore',
            'female_percentage': '% Femmine',
            'fragility_index': 'Indice Fragilità'
        },
        center={"lat": 44.492, "lon": 11.341},
        map_style="carto-positron",
        zoom=11.5,
        opacity=0.5,
        color_continuous_scale='Turbo',
        range_color=range_color,
    )

    fig.update_traces(
        hovertemplate="""
        <b>%{customdata[0]}</b><br>
        Valore: %{customdata[1]}<br>
        Nuovo_valore: %{customdata[2]}<br>
        %Femmine: %{customdata[3]:.2f}<br>
        Indice Fragilita: %{customdata[4]:.2f}<br>
        <extra></extra>
        """,
        customdata=gdf_areas[[
            "area_statistica_norm", "value_rounded",
            "color_value_rounded", "female_percentage", "fragility_index"
        ]].values
    )

    fig.update_layout(
        autosize=False, width=800, height=600,
        coloraxis_colorbar_title_text=None,
    )

    return fig



def compute_kpis(m, evals):
    return {
        'Base inflow [veh/day]': int(evals[m.I_total_base_inflow].mean()),
        'Mode-shifted inflow [veh/day]': int(evals[m.I_total_mode_shifted].mean()),
        'Lost inflow [veh/day]': int(evals[m.I_total_lost].mean()),
        'Modified inflow [veh/day]': int(evals[m.I_total_modified_inflow].mean()),
        'Time-shifted inflow [veh/day]': int(evals[m.I_total_time_shifted].mean()),
        'Paying inflow [veh/day]': int(evals[m.I_total_paying].mean()) if evals[m.I_modified_avg_cost_per_payers].mean() > 0 else 0,
        'Collected fees [€/day]': int(evals[m.I_total_paid].mean()),
        'Emissions [NOx gr/day]': int(evals[m.I_total_modified_emissions].mean()),
        'Emissions difference [NOx gr/day]': int(evals[m.I_total_emissions].mean()) - int(evals[m.I_total_modified_emissions].mean())
    }


def roundup(val):
    v = val * 1.4
    l = math.floor(math.log10(v * 1.3))
    return round(v / 10 ** l) * 10 ** l


if __name__ == "__main__":
    m = Model()

    print(f"Evaluating the model using:\n - {DECISION_STRATEGY} decision-making strategy,\n - {TIME_SHIFT_STRATEGY} time-shifting for anticipating and postponing,\n - {TRAFFIC_COMPUTATION_MODE} method for traffic computation,\n - {MODAL_SHIFT_OPTION} modal shift.")

    subs = m.evaluate(10)

    # ZONE = [14]
    # ZONE = [14, 15]
    # ZONE = 0
    ZONE = 10
    # ZONE = -1
    # In case ZONE == -1 define also DELTA and TIME
    # DELTA = True
    DELTA = False
    # TIME = (to_time(0), to_time(3600 * 24))
    TIME = to_time(3600 * 6)

    if ZONE == 0:
        plot_field_graph(subs[m.I_modified_inflow],
                         horizontal_label="Time", vertical_label="Flow (vehicles/hour)",
                         # vertical_size=1250,
                         vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                         reference_line=subs[m.TS_inflow][0])
        plt.show()
        plot_field_graph(subs[m.I_modified_traffic],
                         horizontal_label="Time", vertical_label="Traffic (circulating vehicles)",
                         # vertical_size=15000,
                         reference_line=subs[m.I_traffic][0])
        plt.show()
        plot_field_graph(subs[m.I_modified_emissions],
                         horizontal_label="Time", vertical_label="Emissions (NOx gr/h)",
                         # vertical_size=3000,
                         vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                         reference_line=subs[m.I_emissions][0])
        plt.show()
        for k, v in compute_kpis(m, subs).items():
            print(f'{k} - {v:,}')

    elif ZONE == -1:
        if DELTA:
            fig = plot_map_graph(subs, m.I_delta_inflow_zone, time=TIME, label="Flow difference (vehicles/h)")
        else:
            fig = plot_map_graph(subs, m.I_modified_inflow_zone, time=TIME, label="Flow (vehicle/h)")
        fig.show()
        if DELTA:
            fig = plot_map_graph(subs, m.I_delta_traffic_zone, time=TIME,
                                 label="Traffic difference (average diff in circulating vehicles)", function=statistics.mean)
        else:
            fig = plot_map_graph(subs, m.I_modified_traffic_zone, time=TIME,
                                 label="Traffic (max circulating vehicles)", function=max)
        fig.show()
        if DELTA:
            fig = plot_map_graph(subs, m.I_delta_emissions_zone, time=TIME,
                                 label="Emission difference (NOx gr/day)")
        else:
            fig = plot_map_graph(subs, m.I_modified_emissions_zone, time=TIME,
                                 label="Emissions (NOx gr/day)")
        fig.show()
        for k, v in compute_kpis(m, subs).items():
            print(f'{k} - {v:,}')

    elif ZONE==10:
        for k, v in compute_kpis(m, subs).items():
            try:
                if len(v) == 1:
                    print(f'{k}: {v}')
                else:
                    print(f'{k} : {[round(float(vv),4) for vv in v],}')
            except TypeError:
                print(f'{k}: {v}')

    else:
        plot_multifield_graph(subs, m.I_modified_inflow_zone, ZONE,
                              horizontal_label="Time", vertical_label="Flow (vehicles/h)",
                              vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                              reference_index=m.TS_inflow_zone)
        plt.show()
        plot_multifield_graph(subs, m.I_modified_traffic_zone, ZONE,
                              horizontal_label="Time", vertical_label="Traffic (circulating vehicles)",
                              reference_index=m.TS_traffic_zone)
        plt.show()
        plot_multifield_graph(subs, m.I_modified_emissions_zone, ZONE,
                              horizontal_label="Time", vertical_label="Emissions (NOx gr/h)",
                              vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                              reference_index=m.I_emissions_zone)
        plt.show()
