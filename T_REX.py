import os
import sys
import random
import json
import numpy as np
from time import time
import sumolib
import pandas as pd
import xml.etree.ElementTree as ET
import csv
import optparse
import traci
from sumolib import checkBinary
import heapq

class Initializer():
    '''
    Initializer is used to initialize the information of incidents.
    The information needed:
    1. Incident edge
    2. Incident position 
    3. Number of blocked lanes
    4. Incident starting time
    5. Incident duration
    '''

    def __init__(self, map_name, run_num, scenario_folder, warm_up_time, end_time=3600, is_random=True, level=2, pre_seed=None):
        # Currently hardcoded values
        # self.slow_zone = 50
        # self.lc_zone = 20
        # self.lc_prob_zone = 70 #170
        self.slow_zone_speed = 2.2 # m/s equivalent to 8 km/h, 5 mph

        self.run_num = run_num
        self.is_incident = False
        self.is_random = is_random
        self.map_name = map_name
        self.warm_up_time = warm_up_time
        self.end_time = end_time
        if pre_seed == None:
            self.random_seed = np.random.randint(2**31 - 1)
        else:
            self.random_seed = pre_seed
        self.scenario_folder = scenario_folder

        self.edge = None
        self.lanes = None
        self.pos = None
        self.start_time = None
        self.start_step = None
        self.duration_time = None
        self.duration_steps = None

        self.net_path = self.scenario_folder
        self.net = sumolib.net.readNet(self.net_path)

        self.level = level
        

        # Using weighted probability for incident edge selection
        # if the map is ingolstadt 
        if self.map_name == 'ingolstadt21':
            self.edge_probabilities = self.load_edge_probability('Ing21_prob.csv')
        if self.map_name == 'ingolstadt7':
            self.edge_probabilities = self.load_edge_probability('Ing7_prob.csv')
        if self.map_name == 'cologne3':
            self.edge_probabilities = self.load_edge_probability('Col3_prob.csv')
        if self.map_name == 'cologne8':
            self.edge_probabilities = self.load_edge_probability('Col8_prob.csv')
        
    def set_incident(self, edge=None, lanes=None, pos=None, start_time=None, duration=None, is_incident=False):
        '''
        This function is used to pre-defined incident based on the input settings.
        '''
        self.is_incident = is_incident
        self.edge = edge
        self.lanes = lanes
        self.pos = pos
        self.start_time = start_time
        self.start_step = start_time
        self.duration_time = duration
        self.duration_steps = duration
    
    def random(self):
        '''
        This function is used to randomly generate incident based on the input settings.
        '''
        np.random.seed(self.random_seed)
        
        if self.map_name in ['ingolstadt21', 'ingolstadt7', 'cologne3', 'cologne8']:
            self.weighted_random_edge()
        else:
            self.random_edge()
        self.random_lanes()
        self.random_pos()
        self.random_time()
        self.random_duration()
        self.is_incident = True

        # print('Incident settings:')
        # print(f'Incident happens at edge {self.edge} at time {self.start_time} lasting for {self.duration_time} seconds')
    
    def load_edge_probability(self, weight_file_name):
        '''
        Load the probability that each edge will have an incident based on history flow data.
        '''
        edge_proabilities = {}
        with open(weight_file_name, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                edge_id = row['Edge ID']
                ratio = float(row['Ratio'])
                edge_proabilities[edge_id] = ratio
        return edge_proabilities
    
    def weighted_random_edge(self):
        '''
        Select edge based on the weighted probability.
        '''
        # Get the list of all edges and remove junctions (starts with ':')
        object_list = traci.edge.getIDList()


        object_list = traci.edge.getIDList()
        
        # Filter out invalid edges (junctions, incomplete edges, and dead-ends)
        valid_edges = [
            edge for edge in object_list 
            if not edge.startswith(':')  # Exclude junctions
            and not any(sub in edge for sub in ['right', 'left', 'bottom', 'top'])  # Exclude incomplete edges
            and len(self.net.getEdge(edge).getOutgoing().keys()) > 0  # Exclude dead-end edges
        ]
        
        # Get the corresponding probabilities (ratios) for the edges
        probabilities = [self.edge_probabilities.get(edge, 0) for edge in valid_edges]
        
        # Normalize probabilities to sum to 1 (in case they don't already sum to 1)
        total_probability = sum(probabilities)
        if total_probability > 0:
            probabilities = [p / total_probability for p in probabilities]
        else:
            raise ValueError("Sum of probabilities is zero. Check your CSV data.")

        # Choose a random edge based on the probability distribution
        self.edge = np.random.choice(valid_edges, p=probabilities)
    

    def random_edge(self):
        """
        Randomly select a valid edge for an incident.
        An edge is valid if:
        1. It is not a junction or incomplete edge.
        2. It has outgoing connections (not a dead-end).
        """
        object_list = traci.edge.getIDList()
        
        # Filter out invalid edges (junctions, incomplete edges, and dead-ends)
        valid_edges = [
            edge for edge in object_list 
            if not edge.startswith(':')  # Exclude junctions
            and not any(sub in edge for sub in ['right', 'left', 'bottom', 'top'])  # Exclude incomplete edges
            and len(self.net.getEdge(edge).getOutgoing().keys()) > 0  # Exclude dead-end edges
        ]

        # Ensure there are valid edges available
        assert valid_edges, f"No valid edges found in the network. Check edge list: {object_list}"

        for edge in valid_edges:
            edge_i_obj = self.net.getEdge(edge)
            downstream_edges_i_obj = list(edge_i_obj.getOutgoing().keys())
            downstream_edges_i = [edge_obj.getID() for edge_obj in downstream_edges_i_obj]
            if len(downstream_edges_i) == 0:
                valid_edges.remove(edge)
                print('remove edge {edge} as it is a dead end')
        
        # Randomly select a valid edge
        self.edge = np.random.choice(valid_edges)

        
        

    def random_lanes(self):
        '''
        Select lanes to be blocked randomly.
        '''
        n_lanes = traci.edge.getLaneNumber(self.edge)
        if self.map_name == 'ingolstadt21':
            # In ingolstadt21, lane 0 is for pedestrian only
            lane_names = np.arange(1, n_lanes) 
        else:
            lane_names = np.arange(0, n_lanes)


        if n_lanes == 1:
            self.lanes = lane_names.tolist()
        else:
            if self.level == 1:
                # Level incident only block partial of edge
                n_blocked_lanes = np.random.randint(1, n_lanes)
            else:
                # probability will block all lanes
                n_blocked_lanes = np.random.randint(1, n_lanes + 1)
            # n_blocked_lanes = np.random.randint(1, n_lanes + 1)
            if np.random.randint(0,2):
                # Block from low
                blocked_lanes = lane_names[:n_blocked_lanes]
            else:
                blocked_lanes = lane_names[-n_blocked_lanes:]

            # For fully randome lanes:
            # blocked_lanes = np.random.choice(lane_names, n_blocked_lanes, replace=False)
            self.lanes = blocked_lanes.tolist()
    
    def random_pos(self):
        '''
        Randomly select position of incident
        '''

        edge_length = traci.lane.getLength(f'{self.edge}_0')
        # Leave room at the end of the edges to avoid bugs
        self.pos = np.random.uniform(10, edge_length - 10)
        return

    def random_time(self):
        '''
        Randomly select starting time of incident
        '''
        self.start_time = np.rint(np.random.uniform(self.warm_up_time, self.end_time - 500)).astype(int)
        self.start_step = self.start_time
        return 
    
    def random_duration(self):
        '''
        Randomly select duration of incident
        '''
        # Incident duration is sampled from exponential distribution (minutes)
        self.duration_time = np.rint(np.random.exponential(1 / 0.029)).astype(int)*60 
        self.duration_steps = self.duration_time
        # print(f'Incident duration: {self.duration_time} seconds')
        return
    
    # def save_incident_information(self, folder_path):
    #     if self.is_incident:
    #         json_str = json.dumps(self.__dict__, default=np_encoder)
    #         file = open(f'{folder_path}/incident_settings.json', 'w+') 
    #         file.write(json_str)
    #     return

    # def load_incident_dict(self, dict):
    #     for k, v in dict.items():
    #         if k not in ['run_num', 'is_random']:
    #             setattr(self, k, v)

class Deployment():
    '''
    The Deployment component is called at every learning step 
    to introduce incidents in each episode. 
    The Deployment consist of five main modules: 
    1. Tringgering: check if an incident should be introduced
    2. Rerouting: simulate the rerouting behavior of vehicles
    3. Lane-changing: simulate the lane-changing behavior of vehicles
    4. Speed-adjustment: simulate the speed of vehicles around incident area
    '''

    def __init__(self, initializer):
        # Get incident information from Initializer
        self.incident_edge = initializer.edge
        self.lanes = initializer.lanes
        self.pos = initializer.pos
        self.start_step = initializer.start_step
        self.duration_steps = initializer.duration_steps
        self.run_num = initializer.run_num
        self.slow_zone = initializer.slow_zone
        self.lc_zone = initializer.lc_zone
        self.lc_prob_zone = initializer.lc_prob_zone
        self.slow_zone_speed = initializer.slow_zone_speed
        self.scenario_folder = initializer.scenario_folder

        # These are set in the traci init
        self.incident_edge_lanes = None        
        self.free_lanes = None
        self.upstream_edges = None
        self.upstream_edges_length = None 
        self.upstream_edges_n_lanes = None
        self.upstream_slow_zone = None
        self.upstream_lc_zone = None
        self.upstream_lc_prob_zone = None

        self.last_lane_change_time = {}

        self.min_lane_change_interval = 50
        self.min_lane_change_interval_upstream = 100


        # ICM parameters
        self.awareness_params = {'pi_news': 0.7, 'pi_on': 0.5, 'T_broadcast': 5, 
                                  'vms_percentage': 0.4  ,'enter_time_vms': 10,
                                 'l2': 200, 'beta_vms': 2, 'pi_online': 0.8, 'enter_time_online': 5,
                                 'sigma': 10, 't_obs_0': 2, 'xi_obs': 0.5}


        self.ICM_params = {'beta_0': -5, 'beta_gain': 2.5, 'beta_loss': 2.5}


        self.net_path = self.scenario_folder
        self.net = sumolib.net.readNet(self.net_path) # Load the network

        # net_path = f'{self.scenario_folder}/grid4x4.net.xml'
        # net = sumolib.net.readNet(net_path)
        # i_edge_obj = net.getEdge(self.incident_edge)

        # Keep track of the vehicles that are adjusted their speed and lane change 
        self.adjusted_vehicles = set()

        self.processed_vehicles = set()
        self.vehicle_ssd = dict()

        # setting for ICM
        self.awareness_vehicles = set()
        self.rerouted_vehicles = set()
        self.checked_edges = {}


    def sim_incident(self, step, reroute=True):
        '''
        Check if incident should be introduced in this step
        '''
        self.speed_adjustment(step)
        # if self.pos < np.max([self.slow_zone, self.lc_prob_zone]):
        #     self.speed_adjustment_upstream(step)
        self.triggering(step)
        # self.simulate_accident_based_on_blocked_lanes(step)
        if reroute:
            self.ICM(step)

    #     return
    def simulate_accident_based_on_blocked_lanes(self, step):
        """
        Simulates different types of accidents based on the number of lanes to be blocked,
        ensuring that multi-lane accidents occur only on lanes of the same edge.

        Args:
            step (int): Current simulation step.
        """
        num_blocked_lanes = 2
        collision_distance = 10  # Distance threshold for collision
        adjacent_collision_distance = 15  # Distance threshold for adjacent lane vehicles
        speed_increase = 50
        global blocked_vehicles  # To track the vehicles involved in the accident

        if step == self.start_step:
            # Get all lanes in the simulation
            lane_ids = traci.lane.getIDList()

            # Group lanes by their edge
            edges = {}
            for lane_id in lane_ids:
                edge_id = traci.lane.getEdgeID(lane_id)
                if edge_id not in edges:
                    edges[edge_id] = []
                edges[edge_id].append(lane_id)

            if num_blocked_lanes == 1:
                # Simulate front and back vehicle collision on a single lane
                for lane_id in lane_ids:
                    vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
                    if len(vehicles_on_lane) >= 2:
                        front_vehicle = vehicles_on_lane[0]
                        back_vehicle = vehicles_on_lane[1]

                        front_pos = traci.vehicle.getLanePosition(front_vehicle)
                        back_pos = traci.vehicle.getLanePosition(back_vehicle)

                        if abs(front_pos - back_pos) <= collision_distance:
                            # Stop both vehicles to simulate a collision
                            traci.vehicle.setSpeed(front_vehicle, 0)
                            traci.vehicle.setDecel(front_vehicle, 9.0)
                            traci.vehicle.setLaneChangeMode(front_vehicle, 0)

                            current_speed = traci.vehicle.getSpeed(back_vehicle)
                            traci.vehicle.setSpeedMode(back_vehicle, 0)
                            traci.vehicle.setSpeed(back_vehicle, current_speed + speed_increase)
                            traci.vehicle.setLaneChangeMode(back_vehicle, 0)

                            blocked_vehicles = [front_vehicle, back_vehicle]
                            print(f"Single-lane accident simulated between {front_vehicle} and {back_vehicle} on {lane_id}")
                            for vehicle in blocked_vehicles:
                                print('Blocked vehicle:', vehicle)
                                traci.vehicle.setType(vehicle, 'CAV2')
                                traci.vehicle.setSpeed(vehicle, 0)
                                traci.vehicle.setLaneChangeMode(vehicle, 0)
                            return

            elif num_blocked_lanes == 2:
                # Simulate a multi-lane collision (two lanes on the same edge)
                for edge_id, lanes in edges.items():
                    if len(lanes) < 2:
                        continue

                    for i, lane_1 in enumerate(lanes):
                        vehicles_on_lane_1 = traci.lane.getLastStepVehicleIDs(lane_1)
                        if len(vehicles_on_lane_1) < 1:
                            continue

                        for lane_2 in lanes[i + 1:]:
                            vehicles_on_lane_2 = traci.lane.getLastStepVehicleIDs(lane_2)
                            if len(vehicles_on_lane_2) < 1:
                                continue

                            for veh_1 in vehicles_on_lane_1:
                                for veh_2 in vehicles_on_lane_2:
                                    pos_1 = traci.vehicle.getLanePosition(veh_1)
                                    pos_2 = traci.vehicle.getLanePosition(veh_2)

                                    if abs(pos_1 - pos_2) <= collision_distance:
                                        # Stop both vehicles to simulate a collision
                                        traci.vehicle.setSpeed(veh_1, 0)
                                        traci.vehicle.setDecel(veh_1, 9.0)
                                        traci.vehicle.setLaneChangeMode(veh_1, 0)

                                        traci.vehicle.setSpeed(veh_2, 0)
                                        traci.vehicle.setDecel(veh_2, 9.0)
                                        traci.vehicle.setLaneChangeMode(veh_2, 0)

                                        current_speed = traci.vehicle.getSpeed(veh_2)
                                        traci.vehicle.setSpeedMode(veh_2, 0)
                                        traci.vehicle.setSpeed(veh_2, current_speed + speed_increase)

                                        blocked_vehicles = [veh_1, veh_2]
                                        print(f"Two-lane accident simulated between {veh_1} and {veh_2} on edge {edge_id}")
                                        for vehicle in blocked_vehicles:
                                            print('Blocked vehicle:', vehicle)
                                            traci.vehicle.setType(vehicle, 'CAV2')
                                            traci.vehicle.setSpeed(vehicle, 0)
                                            traci.vehicle.setLaneChangeMode(vehicle, 0)
                                        return

            elif num_blocked_lanes == 3:
                # Simulate multi-lane collisions (two lanes) and block a vehicle on a third lane
                for edge_id, lanes in edges.items():
                    if len(lanes) < 3:
                        continue

                    for i, lane_1 in enumerate(lanes):
                        vehicles_on_lane_1 = traci.lane.getLastStepVehicleIDs(lane_1)
                        if len(vehicles_on_lane_1) < 1:
                            continue

                        for lane_2 in lanes[i + 1:]:
                            vehicles_on_lane_2 = traci.lane.getLastStepVehicleIDs(lane_2)
                            if len(vehicles_on_lane_2) < 1:
                                continue

                            for veh_1 in vehicles_on_lane_1:
                                for veh_2 in vehicles_on_lane_2:
                                    pos_1 = traci.vehicle.getLanePosition(veh_1)
                                    pos_2 = traci.vehicle.getLanePosition(veh_2)

                                    if abs(pos_1 - pos_2) <= collision_distance:
                                        # Stop both vehicles to simulate a collision
                                        traci.vehicle.setSpeed(veh_1, 0)
                                        traci.vehicle.setDecel(veh_1, 9.0)
                                        traci.vehicle.setLaneChangeMode(veh_1, 0)

                                        traci.vehicle.setSpeed(veh_2, 0)
                                        traci.vehicle.setDecel(veh_2, 9.0)
                                        traci.vehicle.setLaneChangeMode(veh_2, 0)

                                        blocked_vehicles = [veh_1, veh_2]

                                        # Block a vehicle in a third lane if close enough
                                        if len(lanes) > i + 2:
                                            lane_3 = lanes[i + 2]
                                            vehicles_on_lane_3 = traci.lane.getLastStepVehicleIDs(lane_3)

                                            if vehicles_on_lane_3:
                                                veh_3 = vehicles_on_lane_3[0]
                                                veh_3_pos = traci.vehicle.getLanePosition(veh_3)

                                                if abs(pos_1 - veh_3_pos) <= adjacent_collision_distance:
                                                    traci.vehicle.setSpeed(veh_3, 0)
                                                    traci.vehicle.setDecel(veh_3, 9.0)
                                                    traci.vehicle.setLaneChangeMode(veh_3, 0)
                                                    blocked_vehicles.append(veh_3)

                                                    print(f"Three-lane accident simulated with vehicles {veh_1}, {veh_2}, and {veh_3}")
                                                    for vehicle in blocked_vehicles:
                                                        print('Blocked vehicle:', vehicle)
                                                        traci.vehicle.setType(vehicle, 'CAV2')
                                                        traci.vehicle.setSpeed(vehicle, 0)
                                                        traci.vehicle.setLaneChangeMode(vehicle, 0)
                                                    return

            # After the blocking duration, restore the vehicles or remove them
            elif step > self.start_step + self.duration_steps:
                for vehicle in blocked_vehicles:
                    if vehicle in traci.vehicle.getIDList():
                        try:
                            traci.vehicle.setLaneChangeMode(vehicle, 1621)
                            traci.vehicle.setSpeedMode(vehicle, 31)
                            traci.vehicle.setSpeed(vehicle, 10)
                        except Exception as e:
                            print(f"Error restoring vehicle {vehicle}: {e}")

                blocked_vehicles.clear()

            return


    def triggering(self, step):
        '''
        Creating blocked lanes for incident
        '''
        if step == self.start_step:
            for lane in self.lanes:
                incident_veh_id = f'incident_veh_{self.incident_edge}_{lane}_{self.pos}'
                incident_route_id = f'incident_route_{self.incident_edge}_{lane}_{step}'

                # Check for vehicles on the incident lane and remove them if necessary
                on_edge = traci.lane.getLastStepVehicleIDs(f"{self.incident_edge}_{lane}")
                if on_edge:
                    veh_pos_on_edge = []
                    for veh in on_edge:
                        veh_pos_on_edge.append(traci.vehicle.getLanePosition(veh))
                    # print((np.abs(veh_pos_on_edge - np.array(self.pos))))
                    if np.min(np.abs(veh_pos_on_edge - np.array(self.pos))) < 7:          # Checking the closest vehicle should be good enough
                        prob_veh = on_edge[np.argmin(np.abs(veh_pos_on_edge - np.array(self.pos)))]
                        print(f"{prob_veh} is too close, removing it")
                        traci.vehicle.remove(prob_veh)
                
                # Create the incident blocking the lane
                print(f"run {self.run_num} step {step} creating block {self.incident_edge}_{lane}_{self.pos}_{self.start_step}")
                traci.route.add(incident_route_id, [self.incident_edge, self.downstream_edges[0]])
                traci.vehicle.add(vehID=incident_veh_id, routeID=incident_route_id, typeID='IC')
            
                traci.vehicle.moveTo(vehID=incident_veh_id, laneID=f'{self.incident_edge}_{lane}', pos=int(self.pos))
                traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0)
                # traci.vehicle.setLaneChangeMode(vehID=incident_veh_id, laneChangeMode=0) # LIBSUMO as TRACI
                traci.vehicle.setLaneChangeMode(vehID=incident_veh_id, lcm=0) # TRACI
        
        elif step > self.start_step and (step-self.start_step)%100==0 and (step-self.start_step) < self.duration_steps: # Starts moving block to avoid time out
            for lane in self.lanes:
                incident_veh_id = f'incident_veh_{self.incident_edge}_{lane}_{self.pos}'
                incident_route_id = f"incident_route_{self.incident_edge}_{lane}_{self.pos}"
                active_vehicles = traci.vehicle.getIDList()
                if incident_veh_id in active_vehicles:
                    traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0.101)

        elif step > self.start_step and (step-self.start_step)%102==0 and (step-self.start_step) < self.duration_steps: # Stops moving block
            for lane in self.lanes:
                incident_veh_id = f'incident_veh_{self.incident_edge}_{lane}_{self.pos}'
                incident_route_id = f"incident_route_{self.incident_edge}_{lane}_{self.pos}"
                active_vehicles = traci.vehicle.getIDList()
                if incident_veh_id in active_vehicles:
                    traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0)

        elif step==(self.start_step+self.duration_steps): # Removes block
            for lane in self.lanes:
                incident_veh_id = f'incident_veh_{self.incident_edge}_{lane}_{self.pos}'
                print(f"run {self.run_num} step {step} removing block {lane}_{self.pos}_{self.start_step}")
                active_vehicles = traci.vehicle.getIDList()
                if incident_veh_id in active_vehicles:
                    traci.vehicle.remove(vehID=incident_veh_id)
                self.remove_speed_limit()
        return
    
    def speed_adjustment(self, step):
        '''
        Logic for slowing down traffic around incident.
        SSD is calculated and lane change parameters + speed are adjusted only once per vehicle.
        '''
        if step >= self.start_step and step <= (self.start_step + self.duration_steps):

            for lane in self.incident_edge_lanes:
                on_edge = traci.lane.getLastStepVehicleIDs(f"{self.incident_edge}_{lane}")
                cars_on_edge = [car for car in on_edge if 'incident' not in car]

                for veh in cars_on_edge:
                    veh_pos_on_edge = traci.vehicle.getLanePosition(veh)
                    veh_speed = traci.vehicle.getSpeed(veh)
                    dist_to_incident = self.pos - veh_pos_on_edge


                    # Calculate SSD only once per vehicle
                    if veh not in self.vehicle_ssd:
                        perception_reaction_time = 2.5
                        deceleration = 3.4
                        perception_reaction_distance = veh_speed * perception_reaction_time
                        braking_distance = (veh_speed ** 2) / (2 * deceleration)
                        SSD = perception_reaction_distance + braking_distance
                        self.vehicle_ssd[veh] = SSD

                    else:
                        SSD = self.vehicle_ssd[veh]

                    if 0 < dist_to_incident < SSD and veh not in self.adjusted_vehicles:
                        if veh in self.adjusted_vehicles:
                            continue
                        # Adjust lane change behavior
                        # print('Adjusting speed and lane change parameters for vehicle:', veh)
                        traci.vehicle.setParameter(veh, "laneChangeModel.lcStrategic", "1")
                        traci.vehicle.setParameter(veh, "laneChangeModel.lcSpeedGain", "1")
                        traci.vehicle.setParameter(veh, "laneChangeModel.lcCooperative", "1")
                        traci.vehicle.setParameter(veh, "laneChangeModel.lcKeepRight", '0')

                        # Reduce speed
                        traci.vehicle.setMaxSpeed(veh, self.slow_zone_speed)

                        # Mark vehicle as adjusted
                        self.adjusted_vehicles.add(veh)

                    elif veh_pos_on_edge > self.pos:
                        if veh in self.processed_vehicles:
                            continue
                        # Set lane change parameters back to normal
                        # print('Restoring speed and lane change parameters for vehicle:', veh)
                        traci.vehicle.setLaneChangeMode(veh, 1621)
                        # Vehicle already passed the incident – restore normal speed
                        traci.vehicle.setMaxSpeed(veh, 55.55)
                        traci.vehicle.setSpeed(veh, -1)
                        self.processed_vehicles.add(veh)  # Also mark it so we skip next time
        

    def speed_adjustment_upstream(self, step):
        for edge in self.upstream_edges:
            if step >= self.start_step and step <= (self.start_step + self.duration_steps):
                for lane in range(self.upstream_edges_n_lanes_dict[edge]):
                    on_edge = traci.lane.getLastStepVehicleIDs(f"{edge}_{lane}")
                    cars_on_edge = [car for car in on_edge if 'incident' not in car]
                    if cars_on_edge:
                        for veh in cars_on_edge:
                            veh_pos_on_edge = traci.vehicle.getLanePosition(veh)
                            dist_to_edge_end =  self.upstream_edges_length_dict[edge] - veh_pos_on_edge
                            if dist_to_edge_end < self.upstream_slow_zone:
                                traci.vehicle.setMaxSpeed(veh, self.slow_zone_speed)

    
    def remove_speed_limit(self):
        for lane in self.incident_edge_lanes:
            on_edge = traci.lane.getLastStepVehicleIDs(f"{self.incident_edge}_{lane}")
            cars_on_edge = [car for car in on_edge if 'incident' not in car]
            for veh in cars_on_edge:
                traci.vehicle.setMaxSpeed(veh, 55.55)
        
        if self.pos < np.max([self.slow_zone, self.lc_prob_zone]):
            for edge in self.upstream_edges:
                for lane in range(self.upstream_edges_n_lanes_dict[edge]):
                    on_edge = traci.lane.getLastStepVehicleIDs(f"{edge}_{lane}")
                    cars_on_edge = [car for car in on_edge if 'incident' not in car]
                    for veh in cars_on_edge:
                        traci.vehicle.setMaxSpeed(veh, 55.55)
                        traci.vehicle.setSpeed(veh, -1)  # Let SUMO handle normal speed

    def ICM(self, step):
        """
        Information Comply Model (ICM) for Rerouting (Kucharski et al., 2019).
        
        - During the incident, checks vehicles with the incident edge in their route.
        - Calculates driver awareness probability.
        - If the driver is aware, evaluates rerouting probability and reroutes if applicable.
        - Updates vehicle types for visualization purposes.
        
        Parameters:
        - step (int): Current simulation step.
        """
        if self.start_step <= step <= (self.start_step + self.duration_steps):
            # Track vehicles that become aware and are rerouted
            # awareness_vehicles = set()
            # rerouted_vehicles = set()

            # Threshold distance to junction for checking awareness
            junction_threshold = 150.0  # Meters from the end of the edge


            # Iterate over all vehicles in the network
            for vehicle_id in traci.vehicle.getIDList():
                # Check if the vehicle has the incident edge in its route
                vehicle_route = traci.vehicle.getRoute(vehicle_id)
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                if self.incident_edge not in vehicle_route:
                    continue  # Skip vehicles unaffected by the incident

                # Skip vehicles that are already on or have passed the incident edge
                incident_index = vehicle_route.index(self.incident_edge)
                current_index = vehicle_route.index(current_edge) if current_edge in vehicle_route else -1

                if current_index >= incident_index:
                    # Vehicle has reached or passed the incident edge
                    continue

                # Step 1: Awareness Evaluation
                lane_id = traci.vehicle.getLaneID(vehicle_id)
                lane_position = traci.vehicle.getLanePosition(vehicle_id)
                lane_length = traci.lane.getLength(lane_id)

                # Check if the vehicle is near the end of the edge (close to a junction)
                if lane_length - lane_position > junction_threshold:
                    continue  # Skip vehicles that are not near the junction

                if current_edge == self.incident_edge:
                    # Skip further processing for vehicles on the incident edge
                    continue
                exclude_substrings = [":", "right", "left", "bottom", "top"]
                if any(substring in current_edge for substring in exclude_substrings):
                    # Skip further processing for this edge as it is junction or incomplete
                    continue

                # Ensure awareness is checked only once per edge
                if vehicle_id not in self.checked_edges:
                    self.checked_edges[vehicle_id] = set()  # Initialize edge set for this vehicle

                if current_edge in self.checked_edges[vehicle_id]:
                    continue  # Awareness already checked for this edge

                # Mark the edge as checked
                self.checked_edges[vehicle_id].add(current_edge)

                if vehicle_id in self.awareness_vehicles:
                    continue # Skip vehicles that are already aware

                awareness_prob = self.calculate_combined_awareness(step/60, vehicle_id)
                # print(f"Checking awareness of vehicle {vehicle_id} at step {step} at edge {current_edge}: {awareness_prob}")
                if np.random.uniform() < awareness_prob:
                    # Mark vehicle as aware
                    self.awareness_vehicles.add(vehicle_id)
                    # print(f"Vehicle {vehicle_id} is aware of the incident at step {step} at edge {current_edge}")
                    
                    # Update vehicle type for visualization (e.g., to 'CAV1')
                    if traci.vehicle.getTypeID(vehicle_id) != 'CAV1':
                        traci.vehicle.setType(vehicle_id, 'CAV1')
                # else:
                #     print(f'Vehicle {vehicle_id} is not aware at edge {current_edge}')

            # Step 2: Rerouting Evaluation
            to_remove = []
            for vehicle_id in self.awareness_vehicles:
                # Check if the vehicle is still in the simulation
                if vehicle_id not in traci.vehicle.getIDList():
                    to_remove.append(vehicle_id)
                    continue  # Skip this vehicle as it has completed its route or left the simulation


                if vehicle_id in self.rerouted_vehicles:
                    continue  # Skip already rerouted vehicles

                # Check if the vehicle still has the incident edge in its route
                vehicle_route = traci.vehicle.getRoute(vehicle_id)
                if self.incident_edge not in vehicle_route:
                    continue


                
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                exclude_substrings = [":", "right", "left", "bottom", "top"]
                # exclude_substrings = [":"]
                if any(substring in current_edge for substring in exclude_substrings):
                    # Skip further processing for this edge as it is junction or incomplete
                    continue

                if current_edge == self.incident_edge:
                    # Skip further processing for vehicles on the incident edge
                    continue

                # Get lane position and lane length
                current_lane = traci.vehicle.getLaneID(vehicle_id)
                vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
                lane_length = traci.lane.getLength(current_lane)

                # # Define the middle range (e.g., between 25% and 75% of the lane length)
                # lower_bound = 0.25 * lane_length
                # upper_bound = 0.75 * lane_length

                # # Check if the vehicle is in the middle range of the lane
                # if not (lower_bound <= vehicle_position <= upper_bound):
                #     continue  # Skip this vehicle as it is too close to the intersection

                # Calculate rerouting probability using the rerouting model
                # print(f"Checking rerouting for vehicle {vehicle_id} at position {vehicle_position} at edge {current_edge}")

                actual_probs = self.get_actual_probs(vehicle_id)
                typical_probs = self.get_typical_probs(vehicle_id)
                arc_costs = self.get_arcs_cost(vehicle_id)
                reroute_prob = self.reroute_model(
                    actual_probs,
                    typical_probs,
                    arc_costs,
                    self.ICM_params['beta_0'],
                    self.ICM_params['beta_gain'],
                    self.ICM_params['beta_loss']
                )
                # print(f"Actual probabilities: {actual_probs}")
                # print(f"Typical probabilities: {typical_probs}")

                # print(f"Rerouting probability for vehicle {vehicle_id} at step {step} at edge {current_edge}: {reroute_prob}")

                # Reroute if the rerouting probability condition is met
                
                if np.random.uniform() < reroute_prob:
                    self.rerouted_vehicles.add(vehicle_id)
                    # print('Route before rerouting:', vehicle_route)
                    accumulated_travel_time_before = 0
                    for edge in vehicle_route:
                        accumulated_travel_time_before += traci.edge.getTraveltime(edge)
                    # print(f"Accumulated travel time before rerouting: {accumulated_travel_time_before}")
                    # Get current edge and destination
                    

                    # traci.vehicle.rerouteTraveltime(vehicle_id, currentTravelTimes=True)

                    
                    # print("Time to passing through incident:", traci.edge.getTraveltime(self.incident_edge))

                    # Three options for rerouting:
                    # 1. Use greedy rerouting
                    # 2. Use A* search for rerouting
                    # 3. Use Dijkstra's algorithm for rerouting
                    self.a_star(vehicle_id)

                    # rerouted_trip = traci.vehicle.getRoute(vehicle_id)
                    # print('Route after rerouting:', rerouted_trip)
                    # accumulated_travel_time_after = 0
                    # for edge in rerouted_trip:
                    #     accumulated_travel_time_after += traci.edge.getTraveltime(edge)
                    # print(f"Accumulated travel time after rerouting: {accumulated_travel_time_after}")

                    # print(f"Vehicle {vehicle_id} rerouted at step {step} at edge {current_edge}")
                # else:
                #     print(f"Vehicle {vehicle_id} did not reroute at step {step} at edge {current_edge}")

                    # Update vehicle type for visualization (e.g., to 'CAV3')
                    if traci.vehicle.getTypeID(vehicle_id) != 'CAV3':
                        traci.vehicle.setType(vehicle_id, 'CAV3')
            for vehicle_id in to_remove:
                self.awareness_vehicles.remove(vehicle_id)


    def heuristic(self, edge_from, edge_to):
        """
        Heuristic function: Euclidean distance between edge_from and edge_to.
        """
        from_coord = edge_from.getFromNode().getCoord()
        to_coord = edge_to.getToNode().getCoord()
        return np.sqrt((from_coord[0] - to_coord[0])**2 + (from_coord[1] - to_coord[1])**2)

    def get_shortest_path(self, current_edge_id, destination_edge_id):
        """
        Compute the shortest path between current_edge and destination_edge 
        based on current travel times using A* Search.
        """
        start_edge = self.net.getEdge(current_edge_id)
        goal_edge = self.net.getEdge(destination_edge_id)

        # Priority queue for A* search
        pq = []
        heapq.heappush(pq, (0, current_edge_id))  # (f_cost, edge ID)

        # Maps to store distances, costs, and predecessors
        g_costs = {current_edge_id: 0}
        predecessors = {}

        while pq:
            _, current_id = heapq.heappop(pq)
            current_edge = self.net.getEdge(current_id)

            # Stop if we reach the goal edge
            if current_id == destination_edge_id:
                break

            # Get outgoing edges
            outgoing_edges_obj = list(current_edge.getOutgoing().keys())
            outgoing_edges = [edge_obj.getID() for edge_obj in outgoing_edges_obj]

            for next_edge_id in outgoing_edges:
                next_edge = self.net.getEdge(next_edge_id)
                travel_time = traci.edge.getTraveltime(next_edge_id)
                tentative_g_cost = g_costs[current_id] + travel_time
                heuristic_cost = self.heuristic(next_edge, goal_edge)
                f_cost = tentative_g_cost + heuristic_cost

                # Update g_cost and predecessors if a better path is found
                if next_edge_id not in g_costs or tentative_g_cost < g_costs[next_edge_id]:
                    g_costs[next_edge_id] = tentative_g_cost
                    predecessors[next_edge_id] = current_id
                    heapq.heappush(pq, (f_cost, next_edge_id))

        # Reconstruct the shortest path from predecessors
        path = []
        edge_id = destination_edge_id
        while edge_id in predecessors:
            path.insert(0, edge_id)
            edge_id = predecessors[edge_id]
        if path:
            path.insert(0, current_edge_id)  # Add the start edge
        return path

    def a_star(self, vehicle_id):
        """
        Compute the advanced route using A* search and assign it to the vehicle.
        """
        current_edge = traci.vehicle.getRoadID(vehicle_id)
        destination_edge = traci.vehicle.getRoute(vehicle_id)[-1]
        
        if current_edge == destination_edge:
            # Vehicle has already reached its destination
            return
        
        # Calculate the shortest path
        new_route = self.get_shortest_path(current_edge, destination_edge)
        
        if new_route:
            # Assign the new route to the vehicle
            traci.vehicle.setRoute(vehicle_id, new_route)
        else:
            print(f"Could not calculate a valid route for vehicle {vehicle_id}.")


    def get_shortest_path(self, current_edge, destination_edge):
        """
        Compute the shortest path between current_edge and destination_edge 
        based on current travel times using Dijkstra's Algorithm.
        """
        # Priority queue for Dijkstra's algorithm
        pq = []
        heapq.heappush(pq, (0, current_edge))  # (cumulative cost, edge ID)
        
        # Maps to store distances and predecessors
        distances = {current_edge: 0}
        predecessors = {}

        while pq:
            current_cost, current = heapq.heappop(pq)
            
            # Stop if we reach the destination edge
            if current == destination_edge:
                break

            # Get outgoing edges
            current_edge_obj = self.net.getEdge(current)
            outgoing_edges_obj = list(current_edge_obj.getOutgoing().keys())
            outgoing_edges = [edge_obj.getID() for edge_obj in outgoing_edges_obj]

            for next_edge in outgoing_edges:
                travel_time = traci.edge.getTraveltime(next_edge)
                new_cost = current_cost + travel_time

                # Update distances and predecessors if a shorter path is found
                if next_edge not in distances or new_cost < distances[next_edge]:
                    distances[next_edge] = new_cost
                    predecessors[next_edge] = current
                    heapq.heappush(pq, (new_cost, next_edge))

        # Reconstruct the shortest path from predecessors
        path = []
        edge = destination_edge
        while edge in predecessors:
            path.insert(0, edge)
            edge = predecessors[edge]
        if path:
            path.insert(0, current_edge)  # Add the start edge
        return path

    def dijkstra(self, vehicle_id):
        """
        Compute the advanced route using shortest path and assign it to the vehicle.
        """
        current_edge = traci.vehicle.getRoadID(vehicle_id)
        destination_edge = traci.vehicle.getRoute(vehicle_id)[-1]
        
        if current_edge == destination_edge:
            # Vehicle has already reached its destination
            return
        
        # Calculate the shortest path
        new_route = self.get_shortest_path(current_edge, destination_edge)
        
        if new_route:
            # Assign the new route to the vehicle
            traci.vehicle.setRoute(vehicle_id, new_route)
        else:
            print(f"Could not calculate a valid route for vehicle {vehicle_id}.")



    def get_best_outgoing_edge(self, current_edge, destination_edge):
        """
        Selects the outgoing edge with the lowest travel time from the current edge.
        If the destination edge is in the outgoing edges, it is selected directly.
        """
        current_edge_obj = self.net.getEdge(current_edge)
        outgoing_edges_obj = list(current_edge_obj.getOutgoing().keys())
        outgoing_edges = [edge_obj.getID() for edge_obj in outgoing_edges_obj]
        
        # If the destination edge is in the outgoing edges, return it directly
        if destination_edge in outgoing_edges:
            return destination_edge
        
        # Calculate travel times for outgoing edges
        travel_times = {edge: traci.edge.getTraveltime(edge) for edge in outgoing_edges}
        
        # Return the edge with the lowest travel time
        if travel_times:
            return min(travel_times, key=travel_times.get)
        return None

    def calculate_full_route(self, current_edge, destination_edge):
        """
        Calculates the full route from the current edge to the destination edge 
        by selecting outgoing edges with the lowest travel time at each step.
        """
        route = [current_edge]

        while current_edge != destination_edge:
            # Get the next edge based on the greedy choice
            next_edge = self.get_best_outgoing_edge(current_edge, destination_edge)
            if not next_edge:
                # No valid outgoing edge found, terminate with the current route
                print(f"No valid outgoing edge from {current_edge}. Stopping route calculation.")
                break
            
            # Add the next edge to the route
            route.append(next_edge)
            
            # If the next edge is the destination, stop
            if next_edge == destination_edge:
                break
            
            # Update the current edge
            current_edge = next_edge
        
        return route

    def greedy_rerouting(self, vehicle_id):
        """
        Assigns the full route calculated by the greedy approach to the vehicle.
        """
        current_edge = traci.vehicle.getRoadID(vehicle_id)
        destination_edge = traci.vehicle.getRoute(vehicle_id)[-1]
        
        if current_edge == destination_edge:
            # Vehicle has already reached its destination
            return
        
        # Calculate the full route from the current edge to the destination edge
        new_route = self.calculate_full_route(current_edge, destination_edge)
        
        if len(new_route) > 1 and destination_edge in new_route:
            # Set the calculated route for the vehicle
            traci.vehicle.setRoute(vehicle_id, new_route)
        else:
            print(f"Could not calculate a valid route to the destination for vehicle {vehicle_id}.")



    def get_arcs_cost(self, vehicle_id):
        
        # Get the upstream edge of the vehicle
        current_edge = traci.vehicle.getRoadID(vehicle_id)

        current_edge_obj = self.net.getEdge(current_edge)

        upstream_edges_obj = list(current_edge_obj.getIncoming().keys())
        upstream_edges = [edge_obj.getID() for edge_obj in upstream_edges_obj]

        if not upstream_edges:
            return []
        
        arc_costs = []
        for edge in upstream_edges:
            travel_time = traci.edge.getTraveltime(edge)
            arc_costs.append(travel_time)
        
        return arc_costs


    def get_arcs_cost(self, vehicle_id):
        """
        Calculate the costs (e.g., travel time) for downstream arcs of the current edge where the vehicle is located.

        Parameters:
        vehicle_id (str): The ID of the vehicle.

        Returns:
        list: A list of travel times (arc costs) for downstream edges.
        """
        # Get the current edge of the vehicle
        current_edge = traci.vehicle.getRoadID(vehicle_id)

        # Retrieve the edge object for the current edge
        current_edge_obj = self.net.getEdge(current_edge)

        # Get the downstream edges (outgoing edges) from the current edge
        downstream_edges_obj = list(current_edge_obj.getOutgoing().keys())
        downstream_edges = [edge_obj.getID() for edge_obj in downstream_edges_obj]

        # If there are no downstream edges, return an empty list
        if not downstream_edges:
            return []

        # Calculate the travel time (arc cost) for each downstream edge
        arc_costs = []
        for edge in downstream_edges:
            travel_time = traci.edge.getTraveltime(edge)  # Get the travel time for the edge
            arc_costs.append(travel_time)

        return arc_costs

    def add_information_noise(self, values, noise_std=0.1):
        """
        Add noise to simulate imperfect information.
        The noisy level is decided based on driver type:
        - 40% of experienced drivers
        - 30% of novice drivers
        - 20% of distracted drivers
        - 10% of CAVs

        Parameters:
        - values: List or array of values to which noise will be added.
        - noise_std: Standard deviation of the noise.

        Returns:
        - noisy_values: List of values with added noise.
        """
        values = np.array(values, dtype=np.float32)
        # Validate input values
        if values.size == 0:
            print('Values:', values)
            raise ValueError("Input 'values' cannot be empty or have zero dimensions.")

        # Sample driver type
        driver_prob = [0.4, 0.3, 0.2, 0.1]
        driver_type = np.random.choice(['experienced', 'novice', 'distracted', 'CAV'], p=driver_prob)
        
        # Adjust noise level based on driver type
        if driver_type == 'experienced':
            noise_std = 0.05
        elif driver_type == 'novice':
            noise_std = 0.1
        elif driver_type == 'distracted':
            noise_std = 0.2
        elif driver_type == 'CAV':
            noise_std = 0.01

        # Generate noise matching the shape of 'values'
        noise = np.random.normal(0, noise_std, size=values.shape)
        
        # Add noise and ensure non-negativity
        noisy_values = np.maximum(values + noise, 0)
        
        # Normalize to maintain probability distribution
        if np.sum(noisy_values) > 0:
            return noisy_values / np.sum(noisy_values)
        else:
            return np.zeros_like(values)  # Return zeros if normalization fails



    def get_actual_probs(self, vehicle_id):
        """
        Calculate probabilities for a vehicle to choose each downstream edge based on current traffic conditions.

        Parameters:
        - vehicle_id (str): The ID of the vehicle.

        Returns:
        - downstream_edges (list): List of downstream edge IDs.
        - probabilities (list): List of probabilities corresponding to each downstream edge.
        """
        # Get the current edge of the vehicle
        current_edge = traci.vehicle.getRoadID(vehicle_id)

        # Retrieve the edge object for the current edge
        current_edge_obj = self.net.getEdge(current_edge)

        # Get downstream edges (outgoing edges)
        downstream_edges_obj = list(current_edge_obj.getOutgoing().keys())
        downstream_edges = [edge_obj.getID() for edge_obj in downstream_edges_obj]

        if not downstream_edges:
            return []  # No downstream edges

        # Calculate scores for downstream edges based on traffic conditions
        scores = []
        for edge in downstream_edges:
            # Current travel time
            travel_time = traci.edge.getTraveltime(edge)
            
            # Free-flow travel time (length / max speed of lane 0)
            free_flow_time = traci.lane.getLength(f'{edge}_0') / max(traci.lane.getMaxSpeed(f'{edge}_0'), 1e-3)
            
            # Congestion factor (lower values indicate more congestion)
            congestion_factor = free_flow_time / max(travel_time, 1e-3)
            
            # Example scoring: prioritize edges with less congestion (higher congestion factor)
            scores.append(congestion_factor)

        # Normalize scores to calculate probabilities
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores] if total_score > 0 else [0] * len(scores)

        return probabilities


    def get_typical_probs(self, vehicle_id):
        """
        Get the downstream edges of the current edge and prepare a binary vector indicating
        if the downstream edge is part of the vehicle's route.

        Parameters:
        - vehicle_id (str): The ID of the vehicle.

        Returns:
        - downstream_edges (list): List of downstream edge IDs.
        - binary_vector (list): Binary vector where 1 indicates the edge is in the vehicle's route.
        """
        # Get the current edge of the vehicle
        current_edge = traci.vehicle.getRoadID(vehicle_id)

        # Retrieve the edge object for the current edge
        current_edge_obj = self.net.getEdge(current_edge)

        # Get the downstream edges (outgoing edges) from the current edge
        downstream_edges_obj = list(current_edge_obj.getOutgoing().keys())
        downstream_edges = [edge_obj.getID() for edge_obj in downstream_edges_obj]

        if not downstream_edges:
            return []  # No downstream edges

        # Get the vehicle's route
        vehicle_route = traci.vehicle.getRoute(vehicle_id)

        # Prepare the binary vector
        binary_vector = [1 if edge in vehicle_route else 0 for edge in downstream_edges]

        return binary_vector

    def calculate_expected_gain(self, actual_probs, typical_probs):
        """
        Calculate the expected gain (Delta p) using cosine similarity.
        
        Parameters:
        - actual_probs: List of actual arc probabilities.
        - typical_probs: List of typical arc probabilities.
        
        Returns:
        - expected_gain: Delta p.
        """
        if len(actual_probs) == 0 or len(typical_probs) == 0:
            return []
        noisy_actual_probs = self.add_information_noise(actual_probs)

        numerator = np.dot(noisy_actual_probs, typical_probs)
        denominator = np.sqrt(np.sum(np.square(noisy_actual_probs))) * np.sqrt(np.sum(np.square(typical_probs)))
        if denominator == 0:
            return 0  # Avoid division by zero
        return 1 - (numerator / denominator)
    
    def calculate_avoided_loss(self, actual_probs, typical_probs, arc_costs):
        """
        Calculate the avoided loss (Delta w).
        
        Parameters:
        - actual_probs: List of actual arc probabilities.
        - typical_probs: List of typical arc probabilities.
        - arc_costs: List of arc costs (minimum expected cost to destination).
        
        Returns:
        - avoided_loss: Delta w.
        """
        if len(actual_probs) == 0 or len(typical_probs) == 0:
            return []
        noisy_actual_probs = self.add_information_noise(actual_probs)

        numerator = np.dot(typical_probs, arc_costs) - np.dot(noisy_actual_probs, arc_costs)
        denominator = np.dot(typical_probs, arc_costs)
        if denominator == 0:
            return 0  # Avoid division by zero
        return numerator / denominator


    def reroute_model(self, actual_probs, typical_probs, arc_costs, beta_0, beta_gain, beta_loss):
        """
        Calculate the rerouting probability (kappa) using the logit model.
        
        Parameters:
        - delta_p: Expected gain (Delta p).
        - delta_w: Avoided loss (Delta w).
        - beta_gain: Sensitivity to expected gain.
        - beta_loss: Sensitivity to avoided loss.
        - beta_0: control general willingness to reroute
        
        Returns:
        - rerouting_probability: Probability of rerouting.
        """

        delta_p = self.calculate_expected_gain(actual_probs, typical_probs)
        delta_w = self.calculate_avoided_loss(actual_probs, typical_probs, arc_costs)
        if delta_p == [] or delta_w == []:
            return 0
        v_reroute = beta_0 + beta_gain * delta_p - beta_loss * delta_w
        reroute_prob = 1 / (1 + np.exp(-v_reroute))
        return reroute_prob

    def calculate_combined_awareness(self, step, vehicle_id):
        """
        Calculate the combined awareness probability for a vehicle on a given edge.

        Parameters:
        - step (int): Current simulation step.
        - awareness_params (dict): Parameters for awareness model, including:
            - pi_news, pi_on, T_broadcast: Parameters for radio news awareness.
            - edge_with_vms, enter_time_vms, l2, beta_vms: Parameters for VMS awareness.
            - pi_online, enter_time_online, sigma: Parameters for online awareness.
            - t_obs_0, xi_obs: Parameters for observation awareness.
        - vehicle_id (str): ID of the vehicle being evaluated.

        Returns:
        - combined_awareness (float): Combined awareness probability (0 to 1).
        """
        tau = step  # Current time in the simulation
        
        # Step 1: Retrieve the current edge and travel time
        current_edge = traci.vehicle.getRoadID(vehicle_id)
        t_a_tau = traci.edge.getTraveltime(current_edge)  # Travel time on edge (minutes)

        # Step 2: Calculate radio news awareness
        news_prob = self.calculate_arc_radio_awareness(
            self.awareness_params['pi_news'],
            self.awareness_params['pi_on'],
            tau,
            t_a_tau,
            self.awareness_params['T_broadcast']
        )

        # Step 3: Calculate VMS awareness
        vms_edge = self.get_vms_edges(self.awareness_params['vms_percentage'])
        # print(f'VMS edges: {vms_edge}')
        delta_vms = 1 if current_edge in vms_edge else 0
        avg_speed = max(traci.edge.getLastStepMeanSpeed(current_edge), 1e-3)  # Avoid division by zero
        dist_ae = self.calculate_cartesian_distance(current_edge)
        
        vms_prob = self.calculate_vms_awareness(
            delta_vms,
            tau,
            self.start_step + self.awareness_params['enter_time_vms'],
            dist_ae,
            avg_speed,
            self.start_step,
            self.start_step + self.duration_steps,
            self.awareness_params['l2'],
            self.awareness_params['beta_vms']
        )

        # Step 4: Calculate online awareness
        online_prob = self.arc_online_awareness(
            self.awareness_params['pi_online'],
            tau,
            self.start_step + self.awareness_params['enter_time_online'],
            self.awareness_params['sigma'],
            t_a_tau
        )

        # Step 5: Calculate observation-based awareness
        # current_edge_obj = self.net.getEdge(current_edge)
        # edge_length = current_edge_obj.getLength()
        edge_length = traci.lane.getLength(f'{current_edge}_0')
        max_speed = max(traci.lane.getMaxSpeed(f'{current_edge}_0'), 1e-3)  # Avoid division by zero
        t_typical = edge_length / max_speed  # Free-flow travel time
        obs_prob = self.calculate_observation_awareness(
            t_a_tau,
            t_typical,
            self.awareness_params['t_obs_0'],
            self.awareness_params['xi_obs']
        )

        # Step 6: Combine probabilities
        combined_awareness = 1 - (1 - news_prob) * (1 - vms_prob) * (1 - online_prob) * (1 - obs_prob)

        return combined_awareness

    def get_vms_edges(self, percentage, vms_seed=42):
        """
        Selects a fixed random subset of edges based on the given percentage and seed.

        Parameters:
            edge_list (list): List of edge IDs.
            percentage (float): Percentage of edges to select (e.g., 0.4 for 40%).
            seed (int): Seed for random number generator to ensure reproducibility.

        Returns:
            list: Fixed random subset of edges.
        """
        edge_list = traci.edge.getIDList()
        valid_edges = [
            edge for edge in edge_list 
            if not edge.startswith(':')  # Exclude junctions
            and not any(sub in edge for sub in ['right', 'left', 'bottom', 'top'])  # Exclude incomplete edges
            and len(self.net.getEdge(edge).getOutgoing()) > 0  # Exclude dead-end edges
        ]
        # Set the random seed for reproducibility
        local_random = random.Random(vms_seed)
        
        # Calculate the number of edges to select
        num_edges_to_select = int(percentage * len(valid_edges))
        
        # Select the edges
        vms_edges = local_random.sample(valid_edges, num_edges_to_select)
        
        return vms_edges

    def calculate_cartesian_distance(self, edge_id):
        '''
        Calculate term l_ae in the VMS awareness formula.
        '''
        # Get the tail node and its position
        # tail_node = traci.edge.getFromNode(edge_id)
        # tail_pos = traci.junction.getPosition(tail_node) #(x,y)

        tail_pos = self.get_tail_position(edge_id)

        # Get the geometry of the incident edge
        incident_pos_carte = self.get_incident_cart_coords()

        # Calcaulte the Cartesian distance
        distance = np.sqrt((incident_pos_carte[0] - tail_pos[0])**2 +
                       (incident_pos_carte[1] - tail_pos[1])**2)
        return distance
    
    def get_incident_cart_coords(self):
        edge = self.net.getEdge(self.incident_edge)
        incident_coords = self.get_incident_coordinates(edge, self.pos)
        return incident_coords

    def get_incident_coordinates(self, edge, incident_distance):
        edge_shape = edge.getShape()  # Get the shape of the edge (list of (x, y) points)
        cumulative_distance = 0

        for i in range(len(edge_shape) - 1):
            # Calculate segment distance
            segment_distance = self.calculate_distance(edge_shape[i], edge_shape[i + 1])

            # Check if the incident lies in this segment
            if cumulative_distance + segment_distance >= incident_distance:
                # Interpolate the position within this segment
                ratio = (incident_distance - cumulative_distance) / segment_distance
                x = edge_shape[i][0] + ratio * (edge_shape[i + 1][0] - edge_shape[i][0])
                y = edge_shape[i][1] + ratio * (edge_shape[i + 1][1] - edge_shape[i][1])
                return (x, y)
            
            # Update the cumulative distance
            cumulative_distance += segment_distance

        # If the incident distance exceeds the edge length, return the last point
        return edge_shape[-1]
    
    # Function to calculate the distance between two points
    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    
    def get_tail_position(self, edge_id):
        edge = self.net.getEdge(edge_id)
        from_node = edge.getFromNode()
        tail_position = from_node.getCoord()
        return tail_position

    
    def calculate_cumulative_radio_awareness(self, pi_news, pi_on, tau, T_broadcast):
        """
        Calculate the cumulative probability of awareness via radio news at time tau.
        
        Parameters:
        - pi_news: Market penetration of radio listeners (0 to 1).
        - pi_on: Probability of a listener noticing the event during a broadcast (0 to 1).
        - tau: Current time in the simulation (minutes).
        - T_broadcast: Broadcast interval (minutes).
        
        Returns:
        - cumulative_awareness: The cumulative probability of awareness at time tau.
        """
        # Calculate number of broadcasts up to time tau
        n_tau = int(np.floor(tau / T_broadcast))
        
        # Sum over broadcasts
        cumulative_awareness = pi_news * pi_on * sum((1 - pi_on)**(i - 1) for i in range(1, n_tau + 1))
        return cumulative_awareness
    

    def calculate_arc_radio_awareness(self, pi_news, pi_on, tau, t_a_tau, T_broadcast):
        """
        Calculate the awareness probability for a vehicle traveling along arc a.
        
        Parameters:
        - pi_news: Market penetration of radio listeners (0 to 1).
        - pi_on: Probability of a listener noticing the event during a broadcast (0 to 1).
        - tau: Entry time of the vehicle onto arc a (minutes).
        - t_a_tau: Travel time on arc a (minutes).
        - T_broadcast: Broadcast interval (minutes).
        
        Returns:
        - arc_awareness: Probability that the vehicle becomes aware while on arc a.
        """
        # Calculate cumulative probabilities at entry and exit times
        entry_awareness = self.calculate_cumulative_radio_awareness(pi_news, pi_on, tau, T_broadcast)
        exit_awareness = self.calculate_cumulative_radio_awareness(pi_news, pi_on, tau + t_a_tau, T_broadcast)
        
        # Arc-level probability is the difference
        arc_awareness = exit_awareness - entry_awareness
        return arc_awareness

    def calculate_vms_awareness(self, delta_vms, tau, tau_vms, dist_ae, avg_speed, tau0_e, tau1_e, l2, beta_vms):
        """
        Calculate the probability of awareness through VMS.

        Parameters:
        - delta_vms: 1 if arc a is equipped with a VMS, 0 otherwise.
        - tau: Current time in the simulation.
        - tau_vms: Time when the event enters the monitoring system.
        - dist_ae: Cartesian distance from arc tail to the event.
        - avg_speed: Average speed on the network (to convert time to distance).
        - tau0_e: Start time of the event.
        - tau1_e: End time of the event.
        - l2: Distance at which the probability is halved.
        - beta_vms: Shape parameter for probability decay.

        Returns:
        - awareness_prob: Probability that a vehicle becomes aware via VMS.
        """
        # Check if the arc has a VMS and if the information is being displayed
        if delta_vms == 0 or tau < tau_vms:
            return 0  # No VMS or information not yet displayed

        # Calculate temporal distance transformed to spatial distance
        temporal_distance = np.abs(np.median([0, tau0_e - tau, tau1_e - tau]))
        spatial_distance = avg_speed * temporal_distance

        # Total distance from arc to event
        total_distance = dist_ae + spatial_distance

        # Calculate the probability using the decay formula
        awareness_prob = 1 / (1 + (total_distance / l2) ** beta_vms)

        return awareness_prob
    

    def cumulative_online_awareness(self, pi_online, tau, tau_online, sigma):
        """
        Calculate the cumulative probability of awareness via online sources at time tau.
        
        Parameters:
        - pi_online: Market penetration of online users (0 to 1).
        - tau: Current time in the simulation.
        - tau_online: Time when the event information is published online.
        - sigma: Spread rate of the information (controls how quickly awareness spreads).
        
        Returns:
        - cumulative_awareness: The cumulative probability of awareness at time tau.
        """
        if tau < tau_online:
            return 0  # No awareness before the event is published
        
        time_difference = tau - tau_online
        cumulative_awareness = pi_online * (1 - np.exp(-time_difference**2 / (2 * sigma**2)))
        return cumulative_awareness

    def arc_online_awareness(self, pi_online, tau, tau_online, sigma, t_a_tau):
        """
        Calculate the arc-level probability of awareness via online sources.
        
        Parameters:
        - pi_online: Market penetration of online users (0 to 1).
        - tau: Entry time of the vehicle onto arc a (minutes).
        - tau_online: Time when the event information is published online.
        - sigma: Spread rate of the information (minutes).
        - t_a_tau: Travel time on arc a (minutes).
        
        Returns:
        - arc_awareness: Probability of awareness while traversing arc a.
        """
        # Cumulative probability at entry and exit times
        entry_awareness = self.cumulative_online_awareness(pi_online, tau, tau_online, sigma)
        exit_awareness = self.cumulative_online_awareness(pi_online, tau + t_a_tau, tau_online, sigma)
        
        # Arc-level probability is the difference
        arc_awareness = exit_awareness - entry_awareness
        return arc_awareness


    def mid(self, a, b, c):
        """Returns the middle value of a, b, and c."""
        return max(a, min(b, c))

    def calculate_observation_awareness(self, t_actual, t_typical, t_obs_0, xi_obs):
        """
        Calculate the awareness probability via observation.
        
        Parameters:
        - t_actual: Actual travel time on arc a (minutes).
        - t_typical: Typical travel time on arc a (minutes).
        - t_obs_0: Minimum delay threshold for awareness.
        - xi_obs: Sensitivity parameter for delay awareness.
        
        Returns:
        - awareness_prob: Awareness probability via observation (0 to 1).
        """
        # Calculate delay
        delay = t_actual - t_typical
        
        # Scale delay and subtract threshold
        scaled_delay = xi_obs * delay
        result = scaled_delay - t_obs_0
        
        # Constrain result between 0 and 1
        awareness_prob = self.mid(0, result, 1)
        
        return awareness_prob

    
    def remove_speed_limit_reroute(self, rerouted_cars):
        # for veh in rerouted_cars:
        #     if not veh.startswith("incident_veh_"):
        #         traci.vehicle.setMaxSpeed(veh, 55.55)
        active_vehicles = traci.vehicle.getIDList()
    
        for veh in rerouted_cars:
            if not veh.startswith("incident_veh_"):
                # Check if the vehicle is still active in the simulation
                if veh in active_vehicles:
                    traci.vehicle.setMaxSpeed(veh, 55.55)

    def catch_teleporting_vehs(self):
        teleported_vehicles = traci.simulation.getStartingTeleportIDList()
        if len(teleported_vehicles) != 0:
            for veh in teleported_vehicles:
                print('Catched teleported {veh}, restoring standard speed')
                # veh_class = traci.vehicle.getVehicleClass(veh)
                traci.vehicle.setMaxSpeed(veh, 55.55)

    def traci_init(self):
        self.incident_edge_lanes = list(range(traci.edge.getLaneNumber(self.incident_edge)))
        self.free_lanes = list(set(self.incident_edge_lanes) - set(self.lanes)) 
        self.calulate_slow_zone()

    def calulate_slow_zone(self):
        # print('Scenario folder:', scenario_folder)
        net_path = self.scenario_folder
        net = sumolib.net.readNet(net_path)
        i_edge_obj = net.getEdge(self.incident_edge)
        upstream_edges_obj = list(i_edge_obj.getIncoming().keys())
        self.upstream_edges = [edge_obj.getID() for edge_obj in upstream_edges_obj]
        self.upstream_edges_length_dict = {edge_obj.getID():edge_obj.getLength() for edge_obj in upstream_edges_obj} 
        self.upstream_edges_n_lanes_dict = {edge_obj.getID():edge_obj.getLaneNumber() for edge_obj in upstream_edges_obj}
        self.upstream_slow_zone = np.abs(self.slow_zone - self.pos)
        self.upstream_lc_zone = np.abs(self.lc_zone - self.pos) 
        self.upstream_lc_prob_zone = np.abs(self.lc_prob_zone - self.pos) 
            
        downstream_edges_obj = list(i_edge_obj.getOutgoing().keys())
        self.downstream_edges = [edge_obj.getID() for edge_obj in downstream_edges_obj]
        assert len(self.downstream_edges) > 0, 'No downstream edges found for the incident edge {}'.format(self.incident_edge)
        return
