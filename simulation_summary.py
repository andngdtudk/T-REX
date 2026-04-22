import os
import xml.etree.ElementTree as ET


_MODE_CAR = "car"
_MODE_BIKE = "bike"
_MODE_PED = "pedestrian"


def _new_mode_bucket(track_loaded=True, track_safety=True):
    return {
        "inserted": 0,
        "loaded": 0 if track_loaded else None,
        "running": None,
        "waiting": None,
        "teleports": 0,
        "collisions": 0,
        "emergency_stops": 0 if track_safety else None,
        "emergency_brakings": 0 if track_safety else None,
    }


def initialize_runtime_counters():
    return {
        _MODE_CAR: _new_mode_bucket(track_loaded=True, track_safety=True),
        _MODE_BIKE: _new_mode_bucket(track_loaded=True, track_safety=True),
        _MODE_PED: _new_mode_bucket(track_loaded=True, track_safety=False),
        "availability": {
            "vehicle_loaded": True,
            "vehicle_departed": True,
            "vehicle_teleports": True,
            "vehicle_collisions": True,
            "vehicle_emergency_stops": True,
            "vehicle_emergency_brakings": True,
            "person_loaded": False,
            "person_departed": True,
            "person_arrived": True,
            "person_waiting_time": True,
        },
    }


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_call(target, method_name):
    method = getattr(target, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


def _safe_call_with_arg(target, method_name, arg):
    method = getattr(target, method_name, None)
    if method is None:
        return None
    try:
        return method(arg)
    except Exception:
        return None


def _detect_vehicle_mode(veh_id, vtype=None, vclass=None):
    tokens = " ".join(
        [
            str(veh_id or "").lower(),
            str(vtype or "").lower(),
            str(vclass or "").lower(),
        ]
    )
    if "bike" in tokens or "bicycle" in tokens or "cycle" in tokens:
        return _MODE_BIKE
    return _MODE_CAR


def _count_vehicle_ids_by_mode(sumo, vehicle_ids):
    out = {_MODE_CAR: 0, _MODE_BIKE: 0}
    if not vehicle_ids:
        return out

    for veh_id in vehicle_ids:
        vtype = _safe_call_with_arg(sumo.vehicle, "getTypeID", veh_id)
        vclass = _safe_call_with_arg(sumo.vehicle, "getVehicleClass", veh_id)
        mode = _detect_vehicle_mode(veh_id, vtype=vtype, vclass=vclass)
        out[mode] += 1
    return out


def update_runtime_counters(sumo, counters):
    sim = sumo.simulation

    loaded_ids = _safe_call(sim, "getLoadedIDList")
    if loaded_ids is None:
        counters["availability"]["vehicle_loaded"] = False
    else:
        loaded_by_mode = _count_vehicle_ids_by_mode(sumo, loaded_ids)
        counters[_MODE_CAR]["loaded"] += loaded_by_mode[_MODE_CAR]
        counters[_MODE_BIKE]["loaded"] += loaded_by_mode[_MODE_BIKE]

    departed_ids = _safe_call(sim, "getDepartedIDList")
    if departed_ids is None:
        counters["availability"]["vehicle_departed"] = False
    else:
        dep_by_mode = _count_vehicle_ids_by_mode(sumo, departed_ids)
        counters[_MODE_CAR]["inserted"] += dep_by_mode[_MODE_CAR]
        counters[_MODE_BIKE]["inserted"] += dep_by_mode[_MODE_BIKE]

    teleported_ids = _safe_call(sim, "getStartingTeleportIDList")
    if teleported_ids is None:
        counters["availability"]["vehicle_teleports"] = False
    else:
        tel_by_mode = _count_vehicle_ids_by_mode(sumo, teleported_ids)
        counters[_MODE_CAR]["teleports"] += tel_by_mode[_MODE_CAR]
        counters[_MODE_BIKE]["teleports"] += tel_by_mode[_MODE_BIKE]

    colliding_ids = _safe_call(sim, "getCollidingVehiclesIDList")
    if colliding_ids is None:
        counters["availability"]["vehicle_collisions"] = False
    else:
        col_by_mode = _count_vehicle_ids_by_mode(sumo, colliding_ids)
        counters[_MODE_CAR]["collisions"] += col_by_mode[_MODE_CAR]
        counters[_MODE_BIKE]["collisions"] += col_by_mode[_MODE_BIKE]

    emergency_stop_ids = _safe_call(sim, "getEmergencyStoppingVehiclesIDList")
    if emergency_stop_ids is None:
        counters["availability"]["vehicle_emergency_stops"] = False
    else:
        stop_by_mode = _count_vehicle_ids_by_mode(sumo, emergency_stop_ids)
        counters[_MODE_CAR]["emergency_stops"] += stop_by_mode[_MODE_CAR]
        counters[_MODE_BIKE]["emergency_stops"] += stop_by_mode[_MODE_BIKE]

    emergency_brake_ids = _safe_call(sim, "getEmergencyBrakingVehiclesIDList")
    if emergency_brake_ids is None:
        counters["availability"]["vehicle_emergency_brakings"] = False
    else:
        brake_by_mode = _count_vehicle_ids_by_mode(sumo, emergency_brake_ids)
        counters[_MODE_CAR]["emergency_brakings"] += brake_by_mode[_MODE_CAR]
        counters[_MODE_BIKE]["emergency_brakings"] += brake_by_mode[_MODE_BIKE]

    person_loaded_ids = _safe_call(sim, "getLoadedPersonIDList")
    if person_loaded_ids is None:
        counters["availability"]["person_loaded"] = False
    elif counters[_MODE_PED]["loaded"] is not None:
        counters[_MODE_PED]["loaded"] += len(person_loaded_ids)

    person_departed_ids = _safe_call(sim, "getDepartedPersonIDList")
    if person_departed_ids is None:
        counters["availability"]["person_departed"] = False
    else:
        counters[_MODE_PED]["inserted"] += len(person_departed_ids)


def finalize_runtime_state(sumo, counters):
    vehicle_ids = _safe_call(sumo.vehicle, "getIDList") or []
    running_counts = _count_vehicle_ids_by_mode(sumo, vehicle_ids)
    counters[_MODE_CAR]["running"] = running_counts[_MODE_CAR]
    counters[_MODE_BIKE]["running"] = running_counts[_MODE_BIKE]

    car_wait, bike_wait = 0, 0
    for veh_id in vehicle_ids:
        waiting_time = _safe_call_with_arg(sumo.vehicle, "getWaitingTime", veh_id)
        if (waiting_time or 0) <= 0:
            continue
        vtype = _safe_call_with_arg(sumo.vehicle, "getTypeID", veh_id)
        vclass = _safe_call_with_arg(sumo.vehicle, "getVehicleClass", veh_id)
        mode = _detect_vehicle_mode(veh_id, vtype=vtype, vclass=vclass)
        if mode == _MODE_BIKE:
            bike_wait += 1
        else:
            car_wait += 1

    counters[_MODE_CAR]["waiting"] = car_wait
    counters[_MODE_BIKE]["waiting"] = bike_wait

    person_ids = _safe_call(sumo.person, "getIDList")
    if person_ids is None:
        counters[_MODE_PED]["running"] = None
        counters[_MODE_PED]["waiting"] = None
    else:
        counters[_MODE_PED]["running"] = len(person_ids)
        person_wait = 0
        person_wait_available = True
        for person_id in person_ids:
            waiting_time = _safe_call_with_arg(sumo.person, "getWaitingTime", person_id)
            if waiting_time is None:
                person_wait_available = False
                break
            if waiting_time > 0:
                person_wait += 1

        if person_wait_available:
            counters[_MODE_PED]["waiting"] = person_wait
        else:
            counters["availability"]["person_waiting_time"] = False
            counters[_MODE_PED]["waiting"] = None


def _avg(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _format_metric(value):
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


def _parse_tripinfo(tripinfo_path):
    if not os.path.exists(tripinfo_path):
        return []

    try:
        root = ET.parse(tripinfo_path).getroot()
    except Exception:
        return []

    rows = []
    for node in root.findall("tripinfo"):
        trip_id = node.attrib.get("id", "")
        vtype = node.attrib.get("vType", "")
        mode = _detect_vehicle_mode(trip_id, vtype=vtype)
        duration = _as_float(node.attrib.get("duration"), None)
        route_length = _as_float(node.attrib.get("routeLength"), None)
        waiting_time = _as_float(node.attrib.get("waitingTime"), None)
        time_loss = _as_float(node.attrib.get("timeLoss"), None)
        depart_delay = _as_float(node.attrib.get("departDelay"), None)
        speed = None
        if duration is not None and duration > 0 and route_length is not None:
            speed = route_length / duration

        rows.append(
            {
                "mode": mode,
                "duration": duration,
                "routeLength": route_length,
                "speed": speed,
                "waitingTime": waiting_time,
                "timeLoss": time_loss,
                "departDelay": depart_delay,
            }
        )
    return rows


def _parse_personinfo(personinfo_path):
    if not os.path.exists(personinfo_path):
        return []

    try:
        root = ET.parse(personinfo_path).getroot()
    except Exception:
        return []

    rows = []
    for pnode in root.findall("personinfo"):
        duration = _as_float(pnode.attrib.get("duration"), None)
        waiting_time = _as_float(pnode.attrib.get("waitingTime"), None)
        time_loss = _as_float(pnode.attrib.get("timeLoss"), None)
        depart_delay = _as_float(pnode.attrib.get("departDelay"), None)

        walk_nodes = pnode.findall("walk")
        route_lengths = []
        walk_waits = []
        walk_losses = []
        for walk in walk_nodes:
            route_len = _as_float(walk.attrib.get("routeLength"), None)
            walk_wait = _as_float(walk.attrib.get("waitingTime"), None)
            walk_loss = _as_float(walk.attrib.get("timeLoss"), None)
            route_lengths.append(route_len)
            walk_waits.append(walk_wait)
            walk_losses.append(walk_loss)

        route_length = None
        valid_route_lengths = [x for x in route_lengths if x is not None and x >= 0]
        if valid_route_lengths:
            route_length = sum(valid_route_lengths)

        if waiting_time is None:
            waiting_time = sum(x for x in walk_waits if x is not None)
        if time_loss is None:
            time_loss = sum(x for x in walk_losses if x is not None)

        speed = None
        if duration is not None and duration > 0 and route_length is not None:
            speed = route_length / duration

        rows.append(
            {
                "duration": duration,
                "routeLength": route_length,
                "speed": speed,
                "waitingTime": waiting_time,
                "timeLoss": time_loss,
                "departDelay": depart_delay,
            }
        )
    return rows


def _rows_for_stats(rows):
    return [r for r in rows if r.get("duration") is not None and r["duration"] >= 0]


def _stats_from_rows(rows):
    valid = _rows_for_stats(rows)
    return {
        "count": len(valid),
        "routeLength": _avg([r.get("routeLength") for r in valid]),
        "speed": _avg([r.get("speed") for r in valid]),
        "duration": _avg([r.get("duration") for r in valid]),
        "waitingTime": _avg([r.get("waitingTime") for r in valid]),
        "timeLoss": _avg([r.get("timeLoss") for r in valid]),
        "departDelay": _avg([r.get("departDelay") for r in valid]),
    }


def _fallback_int(value, fallback):
    if value is None:
        return fallback
    return value


def _compute_mode_sim_data(mode, rows, counters):
    data = counters.get(mode, {})
    inserted = data.get("inserted")
    loaded = data.get("loaded")

    if inserted in (None, 0) and rows is not None:
        inserted = len(rows)
    if loaded is None:
        loaded = inserted

    out = {
        "inserted": inserted,
        "loaded": loaded,
        "running": data.get("running"),
        "waiting": data.get("waiting"),
        "teleports": data.get("teleports"),
        "collisions": data.get("collisions"),
        "emergency_stops": data.get("emergency_stops"),
        "emergency_brakings": data.get("emergency_brakings"),
    }

    # For pedestrians, only use fields that are available.
    if mode == _MODE_PED:
        avail = counters.get("availability", {})
        if not avail.get("person_loaded", False):
            out["loaded"] = None
        # Collisions and teleports for pedestrians are generally not exposed.
        out["teleports"] = None
        out["collisions"] = None
        out["emergency_stops"] = None
        out["emergency_brakings"] = None

    return out


def _print_section(label, sim_data, stats, include_safety):
    print(f"{label} simulation data:")
    print(f" Inserted / Loaded: {_format_metric(sim_data.get('inserted'))} / {_format_metric(sim_data.get('loaded'))}")
    if sim_data.get("running") is not None:
        print(f" Running: {_format_metric(sim_data.get('running'))}")
    if sim_data.get("waiting") is not None:
        print(f" Waiting: {_format_metric(sim_data.get('waiting'))}")

    teleports = sim_data.get("teleports")
    collisions = sim_data.get("collisions")
    if teleports is not None or collisions is not None:
        print(f" Teleports / Collisions: {_format_metric(teleports)} / {_format_metric(collisions)}")

    if include_safety:
        print(f" Emergency Stops: {_format_metric(sim_data.get('emergency_stops'))}")
        print(f" Emergency Brakings: {_format_metric(sim_data.get('emergency_brakings'))}")

    print(f"{label} statistics (avg of {stats['count']}):")
    print(f" RouteLength: {_format_metric(stats['routeLength'])}")
    print(f" Speed: {_format_metric(stats['speed'])}")
    print(f" Duration: {_format_metric(stats['duration'])}")
    print(f" WaitingTime: {_format_metric(stats['waitingTime'])}")
    print(f" TimeLoss: {_format_metric(stats['timeLoss'])}")
    print(f" DepartDelay: {_format_metric(stats['departDelay'])}")


def print_grouped_mode_summary(log_dir, connection_name, run, counters):
    if run <= 0:
        return

    run_dir = os.path.join(log_dir, connection_name)
    tripinfo_path = os.path.join(run_dir, f"tripinfo_{run}.xml")
    personinfo_path = os.path.join(run_dir, f"personinfo_{run}.xml")

    trip_rows = _parse_tripinfo(tripinfo_path)
    car_rows = [r for r in trip_rows if r.get("mode") == _MODE_CAR]
    bike_rows = [r for r in trip_rows if r.get("mode") == _MODE_BIKE]
    ped_rows = _parse_personinfo(personinfo_path)

    car_sim = _compute_mode_sim_data(_MODE_CAR, car_rows, counters)
    bike_sim = _compute_mode_sim_data(_MODE_BIKE, bike_rows, counters)
    ped_sim = _compute_mode_sim_data(_MODE_PED, ped_rows, counters)

    car_stats = _stats_from_rows(car_rows)
    bike_stats = _stats_from_rows(bike_rows)
    ped_stats = _stats_from_rows(ped_rows)

    print("Car results")
    _print_section("Car", car_sim, car_stats, include_safety=True)

    print("Bike results")
    _print_section("Bike", bike_sim, bike_stats, include_safety=True)

    print("Pedestrian results")
    _print_section("Pedestrian", ped_sim, ped_stats, include_safety=False)
