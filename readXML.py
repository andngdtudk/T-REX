import os
import xml.etree.ElementTree as ET

import numpy as np
import sys
from TREX_comp.config.map_config import map_configs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# log_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'results' + os.sep)
# log_dir = os.path.join(os.path.dirname(os.getcwd()), 'results_s2' + os.sep)

PATHNAME = 'results'

log_dir = os.path.join(os.getcwd(), PATHNAME + os.sep)


# log_dir = '/mnt/raid/andng_backup/results_ic2' + os.sep

env_base = 'environments'+os.sep
# env_base = 'RESCO_main'+os.sep+'environments'+os.sep

names = [folder for folder in next(os.walk(log_dir))[1]]

metrics = ['timeLoss', 'duration', 'waitingTime']

# metrics = ['duration']

# TODO make multimodal
for metric in metrics:
    output_file = 'avg_{}.py'.format(metric)
    metric_var = {
        'timeLoss': 'delays',
        'duration': 'durations',
        'waitingTime': 'waiting',
    }
    run_avg = dict()

    for name in names:
        split_name = name.split('-')
        print(split_name)
        map_name = split_name[2]
        average_per_episode = []
        for i in range(1, 10000):
            trip_file_name = log_dir+name + os.sep + 'tripinfo_'+str(i)+'.xml'
            if not os.path.exists(trip_file_name):
                print('No '+trip_file_name)
                break
            try:
                tree = ET.parse(trip_file_name)
                root = tree.getroot()
                num_trips, total = 0, 0.0
                last_departure_time_actual = 0.0
                last_depart_id = ''
                for child in root:
                    try:
                        id_value = child.attrib.get('id', '')
                        if not id_value.startswith('car'):
                            continue
                        num_trips += 1
                        total += float(child.attrib[metric])
                        if metric == 'timeLoss':
                            total += float(child.attrib['departDelay'])
                            depart_time = float(child.attrib['depart'])
                            if depart_time > last_departure_time_actual:
                                last_departure_time_actual = depart_time
                                last_depart_id = id_value
                    except Exception as e:
                        #raise e
                        break
                # route_file_name = env_base + map_name + os.sep + map_name + os.sep + map_name + '_' + str(i) + '.rou.xml'
                route_file_candidates = [
                    os.path.join(env_base, map_name, 'routes_car.rou.xml'),
                ]
                route_file_candidates = [path for path in route_file_candidates if os.path.exists(path)]
                if not route_file_candidates:
                    route_file_candidates = [os.path.join(env_base, map_name, map_name + '.rou.xml')]

                if metric == 'timeLoss':    # Calc. departure delays
                    depart_by_id = {}
                    all_depart_times = []
                    for route_file_name in route_file_candidates:
                        tree = ET.parse(route_file_name)
                        root = tree.getroot()
                        for child in root:
                            if child.tag != 'vehicle':
                                continue
                            id_value = child.attrib.get('id', '')
                            if not id_value.startswith('car'):
                                continue
                            # Exclude IDs with the form "incident_veh_xxx"
                            if id_value.startswith("incident_veh_"):
                                print('last_depart_id:', last_depart_id)
                                print('Skipping', id_value)
                                continue
                            depart_time = float(child.attrib['depart'])
                            depart_by_id[id_value] = depart_time
                            all_depart_times.append(depart_time)

                    last_departure_time = depart_by_id.get(last_depart_id)
                    if last_departure_time is None:
                        print('Wrong trip file', trip_file_name, 'route file(s):', ', '.join(route_file_candidates))
                        # Fallback to actual departure time to avoid crashing on multimodal runs.
                        last_departure_time = last_departure_time_actual

                    never_departed = []
                    for depart_time in all_depart_times:
                        if depart_time > last_departure_time:
                            never_departed.append(depart_time)
                    never_departed = np.asarray(never_departed)
                    never_departed_delay = np.sum(float(map_configs[map_name]['end_time']) - never_departed)
                    total += never_departed_delay
                    num_trips += len(never_departed)

                average = total / num_trips
                average_per_episode.append(average)
            except ET.ParseError as e:
                #raise e
                break

        run_name = split_name[0]+' '+split_name[2]+' '+split_name[3]+' '+split_name[4]+' '+split_name[5]
        average_per_episode = np.asarray(average_per_episode)

        if run_name in run_avg:
            run_avg[run_name].append(average_per_episode)
        else:
            run_avg[run_name] = [average_per_episode]


    alg_res = []
    alg_name = []
    for run_name in run_avg:
        list_runs = run_avg[run_name]
        min_len = min([len(run) for run in list_runs])
        list_runs = [run[:min_len] for run in list_runs]
        avg_delays = np.sum(list_runs, 0)/len(list_runs)
        err = np.std(list_runs, axis=0)

        alg_name.append(run_name)
        alg_res.append(avg_delays)

        alg_name.append(run_name+'_yerr')
        alg_res.append(err)

        plt.title(run_name)
        plt.plot(avg_delays)
        plt.show()


    np.set_printoptions(threshold=sys.maxsize)
    with open(output_file, 'w') as out:
        var_name = metric_var.get(metric, 'metrics')
        out.write(var_name + " = {\n")
        for i, res in enumerate(alg_res):
            out.write("'{}': {},\n".format(alg_name[i], res.tolist()))
        out.write("}\n")