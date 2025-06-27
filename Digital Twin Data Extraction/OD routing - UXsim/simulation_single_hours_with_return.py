import gc
import os
import sys
import time
from datetime import datetime
sys.path.append(os.path.join(os.path.abspath("../.."), "functions"))

import uxsimulator.sim

# Bool, wether to print updated status or not
verbose = True

# Simulation time for each hour, expressed in seconds
total_simulation_time = 60*60

# Demand threshold and size of the platoons
demand_threshold = 10

# Weights to distribute the daily demand between hours depending on the inflow
list_inflow_weights = [0.01144457, 0.00486095, 0.00243493, 0.00260924, 0.00457556,
       0.01202913, 0.03529495, 0.06563432, 0.07102766, 0.05941337,
       0.05585619, 0.05693654, 0.05517072, 0.05273153, 0.05359615,
       0.05863698, 0.07322648, 0.08225498, 0.07508405, 0.05935842,
       0.03910813, 0.02533367, 0.02307903, 0.02030244]

# Weights to distribute the daily demand between hours depending on the traffic
list_traffic_weights = [0.01493109, 0.00666581, 0.00333917, 0.00236107, 0.00377171,
       0.00875083, 0.02261428, 0.05682909, 0.074271  , 0.06475755,
       0.05954089, 0.05915686, 0.05813329, 0.05474482, 0.05510267,
       0.05692899, 0.0654243 , 0.07393531, 0.07157945, 0.0633979 ,
       0.04717396, 0.02894771, 0.02392624, 0.02371601] # ref

# List of the hours that will be simulated separately and list of the simulation seeds
total_hours = 24
first_hour = 3
total_seeds = 5
list_hours = [i for i in range(first_hour, total_hours)] + [i for i in range(first_hour)]
list_seeds = [i for i in range(total_seeds)]

# Input and output file names
namefile_nodes = "results/nodes_v7.csv"
namefile_edges = "results/edges_v7.csv"
namefile_in_demand = "results/flows_in_v7.csv"
namefile_from_in_demand = "results/flows_from_in_v7.csv"
namefile_to_in_demand = "results/flows_to_in_v7.csv"

namefile_output_edges=f"results/UXsim_links/AreaVerde_links_with_return_v5"
namefile_output_zones=f"results/UXsim_vehicles/AreaVerde_vehicles_with_return_v5"

# Main cycle of the simulation
for i_seed in list_seeds:
    uxsimulator.sim.vprint(text=f"========================== SEED {i_seed} ============================", verbose=verbose)
    start_seed_time = time.time()
    start_seed_dt = datetime.now()

    for hour in list_hours:
        uxsimulator.sim.vprint(text=f"==================== HOURS from_{hour}_to_{hour+1} ======================", verbose=verbose)
        start_hour_time = time.time()
        start_hour_dt = datetime.now()

        W = uxsimulator.sim.create_static_scenario(
            seed=i_seed,
            deltan=demand_threshold,
            total_simulation_time=total_simulation_time,
            namefile_nodes=namefile_nodes,
            namefile_edges=namefile_edges,
            namefile_output_edges=namefile_output_edges,
            namefile_output_zones=namefile_output_zones,
            verbose=verbose
        )

        if len(list_hours) > 1 and hour != list_hours[0]:
            name_prev_iter = f"from_23_to_24_seed_{i_seed}" if hour == 0 else f"from_{hour-1}_to_{hour}_seed_{i_seed}"
            W = uxsimulator.sim.add_hourly_remaining_demand_to_scenario(
                W=W, 
                name_iter=name_prev_iter, 
                namefile_vehicles=namefile_output_zones,
                verbose=verbose
            )

        W = uxsimulator.sim.add_hourly_demand_to_scenario(
            W=W,
            namefile_demand_list=[namefile_in_demand, namefile_from_in_demand, namefile_to_in_demand],
            weights_demand_list=[list_inflow_weights, list_traffic_weights],
            list_hours=list_hours,
            this_hour=hour,
            deltan=demand_threshold,
            verbose=verbose
        )

        W = uxsimulator.sim.execute(
            W=W, 
            duration=60*60,
            verbose=verbose
        )

        uxsimulator.sim.print_analytics(W=W, verbose=verbose)

        uxsimulator.sim.save(
            W=W, 
            name_iter=f"from_{hour}_to_{hour+1}_seed_{i_seed}", 
            namefile_edges=namefile_output_edges,
            namefile_zones=namefile_output_zones,
            verbose=verbose
        )

        del W
        gc.collect()

        end_hour_time = time.time()
        uxsimulator.sim.vprint(text=f"HOUR -- Starting time: {start_hour_dt}; Ending time: {datetime.now()}; Duration: {end_hour_time - start_hour_time:.2f}", verbose=verbose)
    end_seed_time = time.time()
    uxsimulator.sim.vprint(text=f"SEED -- Starting time: {start_seed_dt}; Ending time: {datetime.now()}; Duration: {end_seed_time - start_seed_time:.2f}",verbose=verbose)
