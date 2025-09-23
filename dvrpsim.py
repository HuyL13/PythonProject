"""
DVRP heuristic simulator (Python)
- Models plants (with docks), orders, vehicles and a simple greedy routing heuristic
- Respects: capacity, time windows (arrival and due), dock concurrency, LIFO stacking (heuristic), service time, travel time
- Not an exact MILP solver — it's a realistic simulator useful for testing and prototyping.

How to use: run `python dvrp_simulation.py` — the script contains an example scenario at the bottom.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import heapq
import math
import random
import time

# -------------------- Data classes --------------------
@dataclass
class Plant:
    id: int
    name: str
    docks: int

@dataclass
class Order:
    id: int
    pickup: int  # plant id
    delivery: int  # plant id
    qty: int
    arrival: float  # tie
    due: float  # til
    picked: bool = False
    delivered: bool = False

@dataclass
class Vehicle:
    id: int
    capacity: int
    start_location: int
    location: int = field(init=False)
    time: float = field(default=0.0)
    load: int = field(default=0)
    stack: List[int] = field(default_factory=list)  # order ids (LIFO)
    route: List[Tuple[int, float]] = field(default_factory=list)  # (plant_id, arrival_time)

    def __post_init__(self):
        self.location = self.start_location

# -------------------- Simulator --------------------
class DVRPSimulator:
    def __init__(self,
                 plants: Dict[int, Plant],
                 orders: Dict[int, Order],
                 vehicles: Dict[int, Vehicle],
                 distances: Dict[Tuple[int,int], float],
                 travel_times: Dict[Tuple[int,int], float],
                 dock_service_time: float = 5.0,  # Tda per vehicle before docking
                 unit_handling_time: float = 1.0,  # w per item
                 lambda_cost: float = 100.0):
        self.plants = plants
        self.orders = orders
        self.vehicles = vehicles
        self.distances = distances
        self.travel_times = travel_times
        self.dock_service_time = dock_service_time
        self.unit_handling_time = unit_handling_time
        self.lambda_cost = lambda_cost

        # Track dock occupancy: map plant_id -> list of (vehicle_id, release_time)
        self.dock_occupancy: Dict[int, List[Tuple[int, float]]] = {pid: [] for pid in plants}

    def _travel_time(self, a: int, b: int) -> float:
        if (a,b) in self.travel_times:
            return self.travel_times[(a,b)]
        if (b,a) in self.travel_times:
            return self.travel_times[(b,a)]
        # fallback: proportional to distance
        return self.distances.get((a,b), self.distances.get((b,a), 1.0)) / 30.0

    def _distance(self, a:int, b:int) -> float:
        return self.distances.get((a,b), self.distances.get((b,a), 0.0))

    def _free_docks(self, plant_id: int, at_time: float) -> int:
        # remove released docks
        occ = self.dock_occupancy[plant_id]
        occ = [o for o in occ if o[1] > at_time]
        self.dock_occupancy[plant_id] = occ
        return max(0, self.plants[plant_id].docks - len(occ))

    def _occupy_dock(self, plant_id: int, vehicle_id: int, until_time: float):
        self.dock_occupancy[plant_id].append((vehicle_id, until_time))

    def simulate(self, until_time: float = 24*60):
        # Simple event-driven simulation:
        # - At each vehicle, choose next feasible action greedily:
        #   prefer delivering top-of-stack if at its delivery location and goods on board,
        #   otherwise drive to nearest pickup that is available and feasible (capacity, LIFO)

        active_vehicles = list(self.vehicles.values())
        # keep iterating until all orders delivered or time limit
        all_orders = self.orders
        t0 = 0.0
        iterations = 0
        while iterations < 10000:
            iterations += 1
            undelivered = [o for o in all_orders.values() if not o.delivered and o.arrival <= until_time]
            if not undelivered:
                break
            progressed = False
            for veh in active_vehicles:
                if veh.time > until_time:
                    continue
                # If top of stack can be delivered at current location
                if veh.stack:
                    top_order = self.orders[veh.stack[-1]]
                    if veh.location == top_order.delivery:
                        # check time window
                        arrival_time = veh.time
                        # check dock availability
                        free = self._free_docks(veh.location, arrival_time)
                        if free <= 0:
                            # wait small time for dock
                            veh.time += 1.0
                            progressed = True
                            continue
                        service = self.dock_service_time + self.unit_handling_time * top_order.qty
                        finish = arrival_time + service
                        # deliver
                        veh.time = finish
                        veh.load -= top_order.qty
                        top_order.delivered = True
                        veh.stack.pop()
                        veh.route.append((veh.location, veh.time))
                        self._occupy_dock(veh.location, veh.id, finish)
                        progressed = True
                        continue
                # otherwise, find feasible pickups (orders that have arrived, not picked, capacity ok, LIFO ok)
                feasible = []
                for o in all_orders.values():
                    if o.picked or o.delivered:
                        continue
                    if o.arrival > veh.time:
                        # vehicle can wait until arrival, allow it
                        earliest_start = o.arrival
                    else:
                        earliest_start = veh.time
                    if veh.load + o.qty > veh.capacity:
                        continue
                    # LIFO heuristic: only pick if delivery due >= top_delivery_due
                    if veh.stack:
                        top_del_due = self.orders[veh.stack[-1]].due
                        if o.due < top_del_due:
                            # picking this order would require delivering earlier than top of stack -> breaks LIFO
                            continue
                    # compute travel time to pickup
                    tt = self._travel_time(veh.location, o.pickup)
                    arr = earliest_start + tt
                    # must reach pickup before its due? Not necessary; but pickup after due means impossible to meet delivery
                    # keep candidate
                    feasible.append((o, arr, tt))
                if feasible:
                    # pick the one with earliest due among feasible, tie-break by nearest arrival
                    feasible.sort(key=lambda x: (x[0].due, x[1]))
                    chosen, arr_time, travel_time = feasible[0]
                    # drive to pickup
                    veh.time = arr_time
                    veh.location = chosen.pickup
                    # wait for dock
                    free = self._free_docks(veh.location, veh.time)
                    if free <= 0:
                        veh.time += 1.0
                        progressed = True
                        continue
                    # service time
                    service = self.dock_service_time + self.unit_handling_time * chosen.qty
                    finish = veh.time + service
                    # load
                    veh.time = finish
                    veh.load += chosen.qty
                    chosen.picked = True
                    veh.stack.append(chosen.id)
                    veh.route.append((veh.location, veh.time))
                    self._occupy_dock(veh.location, veh.id, finish)
                    progressed = True
                    continue
                # If nothing feasible, maybe travel to nearest delivery (top of stack) even if not at same location
                if veh.stack:
                    top = self.orders[veh.stack[-1]]
                    if veh.location != top.delivery:
                        tt = self._travel_time(veh.location, top.delivery)
                        veh.time += tt
                        veh.location = top.delivery
                        progressed = True
                        continue
                # If still nothing, advance time slightly
                veh.time += 1.0
                progressed = True
            if not progressed:
                break
        # End simulation loop
        # compute objective
        total_late = 0.0
        total_distance = 0.0
        for o in all_orders.values():
            # compute actual delivery time by scanning vehicle routes? We tracked delivered flag and veh.route arrivals
            # For simplicity, we'll estimate delivery time as the time when delivered flag set (not stored); we could augment Order with delivered_time
            pass
        # To capture delivery times we should have saved them
        # Augment: recompute delivery times by running through vehicles' simulated events stored in route - but we have not stored per-order times.
        # We'll modify above to set order.delivered_time when delivered. For now, recompute by adding that field.

    # --------------- Improved simulate (with delivered_time tracking) ----------------
    def simulate_with_tracking(self, until_time: float = 24*60):
        # reset
        for o in self.orders.values():
            o.picked = False
            o.delivered = False
            if hasattr(o, 'delivered_time'):
                delattr(o, 'delivered_time')
        for v in self.vehicles.values():
            v.location = v.start_location
            v.time = 0.0
            v.load = 0
            v.stack = []
            v.route = []

        self.dock_occupancy = {pid: [] for pid in self.plants}

        iterations = 0
        while iterations < 20000:
            iterations += 1
            undelivered = [o for o in self.orders.values() if not o.delivered and o.arrival <= until_time]
            if not undelivered:
                break
            progressed = False
            # iterate vehicles sorted by their current time (earliest first)
            vehs = sorted(self.vehicles.values(), key=lambda vv: vv.time)
            for veh in vehs:
                if veh.time > until_time:
                    continue
                # deliver top if at location
                if veh.stack:
                    top_o = self.orders[veh.stack[-1]]
                    if veh.location == top_o.delivery:
                        arrival_time = veh.time
                        free = self._free_docks(veh.location, arrival_time)
                        if free <= 0:
                            # wait
                            veh.time += 1.0
                            progressed = True
                            continue
                        service = self.dock_service_time + self.unit_handling_time * top_o.qty
                        finish = arrival_time + service
                        veh.time = finish
                        veh.load -= top_o.qty
                        top_o.delivered = True
                        setattr(top_o, 'delivered_time', finish)
                        veh.stack.pop()
                        veh.route.append((veh.location, veh.time))
                        self._occupy_dock(veh.location, veh.id, finish)
                        progressed = True
                        continue
                # otherwise find feasible pickups
                feasible = []
                for o in self.orders.values():
                    if o.picked or o.delivered:
                        continue
                    if o.arrival > veh.time:
                        earliest = o.arrival
                    else:
                        earliest = veh.time
                    if veh.load + o.qty > veh.capacity:
                        continue
                    if veh.stack:
                        top_due = self.orders[veh.stack[-1]].due
                        if o.due < top_due:
                            continue
                    tt = self._travel_time(veh.location, o.pickup)
                    arr = earliest + tt
                    feasible.append((o, arr, tt))
                if feasible:
                    feasible.sort(key=lambda x: (x[0].due, x[1]))
                    o, arr, tt = feasible[0]
                    # drive
                    veh.time = arr
                    veh.location = o.pickup
                    free = self._free_docks(veh.location, veh.time)
                    if free <= 0:
                        veh.time += 1.0
                        progressed = True
                        continue
                    service = self.dock_service_time + self.unit_handling_time * o.qty
                    finish = veh.time + service
                    veh.time = finish
                    veh.load += o.qty
                    o.picked = True
                    veh.stack.append(o.id)
                    veh.route.append((veh.location, veh.time))
                    self._occupy_dock(veh.location, veh.id, finish)
                    progressed = True
                    continue
                # move toward next top delivery if exists
                if veh.stack:
                    top = self.orders[veh.stack[-1]]
                    if veh.location != top.delivery:
                        tt = self._travel_time(veh.location, top.delivery)
                        veh.time += tt
                        veh.location = top.delivery
                        progressed = True
                        continue
                # else idle small time
                veh.time += 1.0
                progressed = True
            if not progressed:
                break

        # compute objectives
        total_late = 0.0
        for o in self.orders.values():
            if not hasattr(o, 'delivered_time'):
                # undelivered -> penalize heavily
                total_late += (o.due * 10.0)
            else:
                late = max(0.0, getattr(o, 'delivered_time') - o.due)
                total_late += late
        # average distance per vehicle
        total_distance = 0.0
        for v in self.vehicles.values():
            dist = 0.0
            prev = v.start_location
            for (loc, _) in v.route:
                dist += self._distance(prev, loc)
                prev = loc
            total_distance += dist
        avg_distance = total_distance / max(1, len(self.vehicles))
        f_obj = self.lambda_cost * total_late + avg_distance
        return {
            'total_late': total_late,
            'avg_distance': avg_distance,
            'objective': f_obj,
            'orders': {oid: {
                'picked': o.picked,
                'delivered': o.delivered,
                'delivered_time': getattr(o, 'delivered_time', None)
            } for oid,o in self.orders.items()},
            'vehicles': {vid: {
                'route': v.route,
                'final_time': v.time,
                'final_location': v.location
            } for vid,v in self.vehicles.items()}
        }

# -------------------- Example scenario & runner --------------------
if __name__ == '__main__':
    # create simple test instance
    plants = {
        1: Plant(1,'P1', docks=2),
        2: Plant(2,'P2', docks=1),
        3: Plant(3,'P3', docks=1),
        0: Plant(0,'Depot', docks=2)
    }
    # distances (symmetric)
    distances = {
        (0,1): 10.0, (0,2): 20.0, (0,3): 25.0,
        (1,2): 12.0, (1,3): 15.0, (2,3): 8.0
    }
    # travel times in minutes (rough)
    travel_times = {k: v/40.0*60.0 for k,v in distances.items()}  # assume avg speed 40 km/h -> minutes

    # create orders
    orders = {
        1: Order(1, pickup=1, delivery=2, qty=2, arrival=10.0, due=180.0),
        2: Order(2, pickup=1, delivery=3, qty=1, arrival=20.0, due=160.0),
        3: Order(3, pickup=2, delivery=3, qty=3, arrival=0.0, due=240.0),
        4: Order(4, pickup=3, delivery=1, qty=1, arrival=30.0, due=200.0),
    }

    vehicles = {
        1: Vehicle(1, capacity=5, start_location=0),
        2: Vehicle(2, capacity=4, start_location=0)
    }

    sim = DVRPSimulator(plants, orders, vehicles, distances, travel_times,
                        dock_service_time=5.0, unit_handling_time=2.0, lambda_cost=50.0)

    res = sim.simulate_with_tracking(until_time=8*60)  # simulate 8 hours
    print('Result summary:')
    print('Total late (minutes):', res['total_late'])
    print('Avg distance per vehicle:', res['avg_distance'])
    print('Objective:', res['objective'])
    for vid,info in res['vehicles'].items():
        print('Vehicle', vid, 'route events:', info['route'], 'final time', info['final_time'])

    print('\nPer-order delivery info:')
    for oid,o in res['orders'].items():
        print('Order', oid, o)
