import uxsim
import pandas as pd
import numpy as np
import os
import gc
import objgraph
import inspect

from uxsim.ResultGUIViewer import ResultGUIViewer
import uxsimulator.analysis.utils
from io_utils import vprint


def create_static_scenario(
    seed: int,
    deltan: int,
    total_simulation_time: int,
    namefile_nodes: str,
    namefile_edges: str,
    namefile_output_edges: str,
    namefile_output_zones: str,
    verbose: bool = False
):
    vprint(text="-------------------- Creating scenario --------------------", verbose=verbose)
    W = uxsim.World(
            name="TEST",
            deltan=deltan,
            tmax=total_simulation_time,  
            print_mode=1 if verbose else 0, 
            save_mode=1, 
            show_mode=0, 
            random_seed=seed,
            show_progress_deltat=60*30,
            duo_update_time=60*15,
            meta_data={"namefile_output1": namefile_output_edges, "namefile_output2": namefile_output_zones},
    )

    vprint(text="Generating nodes", verbose=verbose)
    with open(namefile_nodes) as f:
        for r in uxsim.csv.reader(f):
            W.addNode(name=r[0], x=float(r[1]), y=float(r[2]))

    vprint(text='Generating links', verbose=verbose)
    with open(namefile_edges) as f:
        for r in uxsim.csv.reader(f):
            W.addLink(r[0], r[1], r[2], length=float(r[3]), free_flow_speed=float(r[4]), number_of_lanes=int(float(r[5])))

    return W


def add_hourly_demand_to_scenario(
    W: uxsim.World,
    namefile_demand_list: list[str],
    weights_demand_list: list,
    list_hours: list,
    this_hour: int,
    deltan: float,
    vehicle_start_time: int = 0,
    vehicle_simulation_time: int = 60*60,
    verbose: bool = False
):
    vprint(text='Generating demand', verbose=verbose)
    attribute = {"added_prev_hour": False, "prev_name": ""}
    vol_tot = 0 # FIXME(araiari): debug only
    vol_tot_red = 0 # FIXME(araiari): debug only
    vol_tot_very_red = 0 # FIXME(araiari): debug only
    vol_or_effettivo = 0 # FIXME(araiari): debug only
    for demand_type in range(3):
        weights_demand = weights_demand_list[0] if demand_type == 2 else weights_demand_list[1]
        with open(namefile_demand_list[demand_type]) as f:
            iter = 0
            for r in uxsim.csv.reader(f):
                iter += 1
                x_orig, y_orig, radious_orig, x_dest, y_dest, radious_dest, volume, *rest = map(float, r)

                vol_tot += volume # FIXME(araiari): debug only
                
                # Check that origin and destination differ #TODO(araiari): add also this flows to the model
                if (x_orig==x_dest) and (y_orig==y_dest) and (radious_orig==radious_dest):
                    continue
                
                vol_tot_red += round(volume/deltan)*deltan # FIXME(araiari): debug only

                # Check that there is enough volume
                if round(volume/deltan) == 0:
                    continue

                # Distribute the vehicle volume per hour
                vol_or_effettivo += round(volume/deltan)*deltan*weights_demand[this_hour] # FIXME(araiari): debug only
                np.random.seed(iter)
                samples = np.random.choice(list_hours, size=round(volume/deltan), p=[weights_demand[i] for i in list_hours])
                counts = np.bincount(samples, minlength=24)
                counts = counts * deltan

                if counts[this_hour] > 0:
                    vol_tot_very_red += counts[this_hour] # FIXME(araiari): debug only
                    W.adddemand_area2area2(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), 
                                        t_start=vehicle_start_time, t_end=vehicle_start_time+vehicle_simulation_time, 
                                        volume=counts[this_hour],attribute=attribute, auto_rename_vehicles=True)

    print(f"vol_totale = {vol_tot} ------ vol_totale_dn = {vol_tot_red}") # FIXME(araiari): debug only
    print(f"vol_orario_desiderato = {vol_or_effettivo}  ------- vol_orario_effettivo = {vol_tot_very_red}") # FIXME(araiari): debug only
    return W


def add_daily_demand_to_scenario(
    W: uxsim.World,
    list_hours: list,
    deltan: int,
    namefile_demand_list: list[str],
    weights_demand_list: list,
    verbose: bool = False
):
    vprint(text='Generating demand', verbose=verbose)

    # vol_tot = 0 # FIXME(araiari): debug only
    # vol_tot_eff = 0 # FIXME(araiari): debug only
    # vol_h_tot = np.zeros(24) # FIXME(araiari): debug only

    attribute = {"added_prev_hour": False, "prev_name": ""}
    for demand_type in range(3):
        weights_demand = weights_demand_list[0] if demand_type == 2 else weights_demand_list[1]
        with open(namefile_demand_list[demand_type]) as f:
            for r in uxsim.csv.reader(f):

                # Read the r-th line of the file 
                x_orig, y_orig, radious_orig, x_dest, y_dest, radious_dest, volume, *rest = map(float, r)

                # Check that the origin and destination are different
                if (x_orig==x_dest) and (y_orig==y_dest) and (radious_orig==radious_dest):
                    continue

                # vol_tot += volume # FIXME(araiari): debug only
                # vol_tot_eff += round(volume/deltan)*deltan # FIXME(araiari): debug only

                # Check that there is enough volume
                if int(volume/deltan) == 0:
                    continue

                # Distribute the vehicle volume per hour
                samples = np.random.choice(list_hours, size=round(volume/deltan), p=[weights_demand[i] for i in list_hours])
                counts = np.bincount(samples, minlength=24)
                counts = counts * deltan

                # Add the vehicle volume to all hours which have enough vehicles (>deltan)
                valid_hours = [hour for hour in list_hours if counts[hour] >= deltan]
                if len(valid_hours) > 0:
                    valid_volumes = [counts[hour] for hour in valid_hours]
                    for hour, volume_w in zip(valid_hours, valid_volumes):
                        h_sim = hour-list_hours[0] if hour>=list_hours[0] else hour-list_hours[0] + max(list_hours)+1 
                        start_second = h_sim*60*60
                        end_second = (h_sim+1)*60*60
                        W.adddemand_area2area2(x_orig, y_orig, radious_orig, x_dest, y_dest, radious_dest, 
                                            t_start=start_second, t_end=end_second, volume=volume_w, 
                                            attribute=attribute, auto_rename_vehicles=True)
                        # vol_h_tot[hour] += volume_w # FIXME(araiari): for debug only

        # print(f"vol_totale = {vol_tot} ------ vol_totale_dn = {vol_tot_eff}") # FIXME(araiari): for debug only
        # print(f"vol_orario = {vol_h_tot}") # FIXME(araiari): for debug only
    return W


def add_hourly_remaining_demand_to_scenario(
    W: uxsim.World, 
    name_iter: str,
    namefile_vehicles: str,
    verbose: bool = False
): 
    vprint(text='Add previous remaining demand', verbose=verbose)
    
    # Read the vehicle file of the hour
    df = pd.read_csv(f"{namefile_vehicles}_{name_iter}.csv", dtype={0: str}).rename(columns={'name': 'vehicle_id'})
    
    # Filter not completed trips only
    df_ended = df[df['link'] == 'trip_end']['vehicle_id'].tolist()
    df = df[~df['vehicle_id'].isin(df_ended)]

    # In case all trips were completed, return the scenario
    if df.shape[0] == 0 or df.empty:
        return W
    
    vprint(text=f"There are {df['vehicle_id'].nunique()} not-completed trips added", verbose=verbose)
    
    # Add vehicles as additional demand 
    for idx, grouped in df.groupby('vehicle_id'):
        orig = grouped["link"].iloc[-1]
        dest = f"{grouped["dest"].iloc[0]}"
        attribute = {"added_prev_hour": True, "prev_name": idx}
        
        if orig == "waiting_at_origin_node":
            orig = f"{grouped["orig"].iloc[0]}"
            W.addVehicle(orig=orig, dest=dest, departure_time=0, 
                        name=str(idx), attribute=attribute, direct_call=False, auto_rename=True)
        elif uxsimulator.analysis.utils.is_valid_string(orig):
            orig = orig.split("_")[0]
            W.addVehicle(orig=orig, dest=dest, departure_time=0, 
                        name=str(idx), attribute=attribute, direct_call=False, auto_rename=True)
    return W


def execute(
    W: uxsim.World,
    duration: int|None = None,
    verbose: bool = False
):
    vprint(text="------------------------ Simulation ------------------------", verbose=verbose)
    W.exec_simulation(duration_t2=duration)

    return W


def save(
    W: uxsim.World,
    name_iter: str,
    namefile_edges: str|None = None,
    namefile_zones: str|None = None,
    save_completed: bool = False,
    verbose: bool = False

):
    vprint(text='------------------------- Saving -------------------------', verbose=verbose)
    if namefile_edges is not None:
        vprint(text=f"Saving relevant results -- traffic in {namefile_edges}_{name_iter}.parquet", verbose=verbose)
        df = W.analyzer.link_to_pandas()
        df.to_parquet(f"{namefile_edges}_{name_iter}.parquet", index=False)
        del df

    if namefile_zones is not None:
        vprint(text=f"Saving relevant results -- vehicles in {namefile_zones}_{name_iter}.parquet", verbose=verbose)

        out = [["name", "dn", "orig", "dest", "t", "link", "x", "s", "v", "attribute"]]
        for veh in W.VEHICLES.values():
            if save_completed and veh in W.VEHICLES_LIVING.values():
                next
            linkname_old = "ImpossibleName"
            for i in range(len(veh.log_t)):
                if veh.log_state[i] in ("wait", "run", "end", "abort"):
                    if veh.log_link[i] != -1:
                        linkname = veh.log_link[i].name
                    else:
                        if veh.log_state[i] == "wait":
                            linkname = "waiting_at_origin_node"
                        elif veh.log_state[i] == "abort":
                            linkname = "trip_aborted"
                        else:
                            linkname = "trip_end"
                    veh_dest_name = None
                    if veh.dest != None:
                        veh_dest_name = veh.dest.name
                    if linkname != linkname_old or i == len(veh.log_t)-1:
                        out.append([str(veh.name), W.DELTAN, veh.orig.name, veh_dest_name, 
                                    veh.log_t[i], linkname, veh.log_x[i], veh.log_s[i], 
                                    veh.log_v[i], veh.attribute])
                    linkname_old = linkname
        df = pd.DataFrame(out[1:], columns=out[0])
        df.to_parquet(f"{namefile_zones}_{name_iter}.parquet", index=False)
        del df

        # # Find a Vehicle object that should be deleted  # FIXME (araiari): debug only
        # sample_vehicle = next((v for k, v in W.VEHICLES.items() if k not in W.VEHICLES_LIVING), None)

        # if sample_vehicle:
        #     # Get its ID and key for reference
        #     vehicle_id = id(sample_vehicle)
        #     vehicle_key = next(k for k, v in W.VEHICLES.items() if id(v) == vehicle_id)
            
        #     print(f"Checking references to Vehicle {vehicle_key} (id: {vehicle_id})")
            
        #     # Show what's referencing this object BEFORE deletion
        #     print("References before deletion:")
        #     objgraph.show_backrefs([sample_vehicle], filename='references_before.png', max_depth=5)
            
        #     remove_keys = [i for i in W.VEHICLES.keys() if i not in W.VEHICLES_LIVING.keys()]
        #     for key in remove_keys:
        #         del W.VEHICLES[key] 
        #     W.VEHICLES = W.VEHICLES_LIVING.copy()
        #     gc.collect()    
        #     vprint(text=f"Indeed now the vehicles in the world are {len(W.VEHICLES.values())}", verbose=verbose)
        
        #     # Find the object again if it still exists
        #     for obj in gc.get_objects():
        #         if id(obj) == vehicle_id:
        #             print(f"Vehicle {vehicle_key} still exists after deletion")
                    
        #             # Show what's referencing this object AFTER deletion
        #             print("References after deletion:")
        #             objgraph.show_backrefs([obj], filename='references_after.png', max_depth=5)
                    
        #             # Get a more detailed count of referrers
        #             print("\nObjects referencing this vehicle:")
        #             for referrer in gc.get_referrers(obj):
        #                 # Try to describe the referrer
        #                 if hasattr(referrer, '__class__'):
        #                     referrer_type = referrer.__class__.__name__
        #                 else:
        #                     referrer_type = type(referrer).__name__
                        
        #                 # If it's a dictionary, try to find which one
        #                 if isinstance(referrer, dict):
        #                     # Try to identify the dictionary
        #                     container_info = "unknown dictionary"
        #                     for name, value in globals().items():
        #                         if value is referrer:
        #                             container_info = f"global dict '{name}'"
        #                             break
                            
        #                     # Attempt to find the key for our vehicle
        #                     for k, v in referrer.items():
        #                         if v is obj:
        #                             container_info += f" with key '{k}'"
        #                             break
                                    
        #                     print(f" - {referrer_type}: {container_info}")
                            
        #                 # Add this where you're printing the referrers
        #                 elif isinstance(referrer, list):
        #                     print(f" - list with {len(referrer)} items")

        #                     print(referrer[164710])
                            
        #                     # Try to identify the list
        #                     for name, value in globals().items():
        #                         if value is referrer:
        #                             print(f"   This appears to be the global list named '{name}'")
                            
        #                     # Print position of our vehicle in the list
        #                     for i, item in enumerate(referrer):
        #                         if item is obj:
        #                             print(f"   Vehicle found at position {i}")
                                    
        #                     # Check if this list belongs to another object
        #                     for container in gc.get_objects():
        #                         for attr_name in dir(container):
        #                             try:
        #                                 if getattr(container, attr_name) is referrer:
        #                                     print(f"   This list belongs to a {type(container).__name__} object as .{attr_name}")
        #                                     break
        #                             except:
        #                                 pass
        #                 else:
        #                     print(f" - {referrer_type}")
                    
        #             break
        #     else:
        #         print(f"Vehicle {vehicle_key} was successfully deleted")                                            

        '''
        # Delete complete vehicles
        if delete_ended:
            vprint(text=f"{len_living} vehicles are kept among the total {len_all} vehicles", verbose=verbose)
            for key in W.VEHICLES.keys():
                if key not in W.VEHICLES_LIVING.keys():
                    del W.VEHICLES[key] 
            W.VEHICLES = W.VEHICLES_LIVING.copy()
            gc.collect()    
            vprint(text=f"Indeed now the vehicles in the world are {len(W.VEHICLES.values())}", verbose=verbose)
        '''

def print_analytics(
    W: uxsim.World,
    verbose: bool = False
):
    if verbose:
        vprint("------------------------- Analysis -------------------------", verbose=verbose)
        W.analyzer.print_simple_stats()


def visualize(
    W: uxsim.World,
    verbose: bool = False
):
    vprint(text='------------------ Visualization ----------------------', verboe=verbose)
    ResultGUIViewer.launch_World_viewer(W)


def online_save_end_vehicles(
    W: uxsim.World
):
    namefile_output = W.meta_data["namefile_output2"]
    vehicles_to_remove = []

    # If the vehicle concluded the trip, save the vehicle data and store its name
    for veh in W.VEHICLES.values():       
        i_last = len(veh.log_t) - 1 
        if veh.log_state[i_last] == "end" and veh.log_link[i_last] == -1: #WHY??
            _save_single_veh(veh, dn=W.DELTAN, namefile_output=namefile_output)
            vehicles_to_remove.append(veh.name)
    
    # Delete complete vehicles
    for veh_name in vehicles_to_remove:
        del W.VEHICLES[veh_name]
    if vehicles_to_remove:
        gc.collect()


def _save_single_veh(
    veh: uxsim.Vehicle, 
    dn: int, 
    namefile_output: str
):
    # Iterates over the vehicle status to process and store them
    out = [["name", "dn", "orig", "dest", "t", "link", "x", "s", "v", "attribute"]]
    linkname_old = "ImpossibleName"
    for i in range(len(veh.log_t)):
        if veh.log_state[i] in ("wait", "run", "end", "abort"):
            if veh.log_link[i] != -1:
                linkname = veh.log_link[i].name
            else:
                if veh.log_state[i] == "wait":
                    linkname = "waiting_at_origin_node"
                elif veh.log_state[i] == "abort":
                    linkname = "trip_aborted"
                else:
                    linkname = "trip_end"
            veh_dest_name = None
            if veh.dest != None:
                veh_dest_name = veh.dest.name
            if linkname != linkname_old or i == len(veh.log_t)-1:
                out.append([veh.name, dn, veh.orig.name, veh_dest_name, 
                            veh.log_t[i], linkname, veh.log_x[i], veh.log_s[i], 
                            veh.log_v[i], veh.attribute])
            linkname_old = linkname
    df = pd.DataFrame(out[1:], columns=out[0])

    # Create a new file if not yet existing
    if not os.path.exists(f"{namefile_output}.csv"):
        df.to_csv(f"{namefile_output}.csv", index=False)
    else:
        # Append to the file otherwise
        df.to_csv(f"{namefile_output}.csv", mode='a', header=False, index=False)
