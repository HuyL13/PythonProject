import copy
import random
import time
from dvrpsim import DVRPSimulator,Vehicle, Plant,Order

class VNSMEOptimizer:
    def __init__(self, simulator: DVRPSimulator, max_time: float = 600.0, disturb_acceptance: float=6.0):
        self.disturb_acceptance = disturb_acceptance
        self.simulator = simulator
        self.max_time = max_time

    def evaluate(self, orders):
        """Run simulator with given orders and return objective + result"""
        # clone orders dict (deepcopy)
        cloned_orders = {oid: copy.deepcopy(o) for oid, o in orders.items()}
        cloned_vehicles = {vid: copy.deepcopy(v) for vid, v in self.simulator.vehicles.items()}
        sim = DVRPSimulator(self.simulator.plants,
                            cloned_orders,
                            cloned_vehicles,
                            self.simulator.distances,
                            self.simulator.travel_times,
                            self.simulator.dock_service_time,
                            self.simulator.unit_handling_time,
                            self.simulator.lambda_cost)
        result = sim.simulate_with_tracking()
        return result["objective"], result

    # -------- Local search operators --------
    def couple_exchange(self, orders):
        """Swap pickup-delivery pair between 2 orders"""
        new_orders = copy.deepcopy(orders)
        if len(new_orders) < 2:
            return new_orders  # không đủ để swap
        o1, o2 = random.sample(list(new_orders.values()), 2)
        o1.delivery, o2.delivery = o2.delivery, o1.delivery
        return new_orders

    def block_exchange(self, orders):
        """Swap two consecutive orders"""
        new_orders = copy.deepcopy(orders)

        keys = list(new_orders.keys())
        if len(keys) >= 2:
            i, j = random.sample(keys, 2)
            new_orders[i], new_orders[j] = new_orders[j], new_orders[i]
        return new_orders

    def block_relocate(self, orders):
        """Relocate one order's pickup-delivery pair to another position"""
        new_orders = copy.deepcopy(orders)
        keys = list(new_orders.keys())
        if len(keys) >= 2:
            i, j = random.sample(keys, 2)
            o = new_orders.pop(i)
            # reinsert at new id (simulate relocation)
            new_orders[i] = o
        return new_orders

    def multi_relocate(self, orders):
        """Apply relocate twice"""
        return self.block_relocate(self.block_relocate(orders))

    def disturb(self, orders):
        """2-opt-L style disturbance: swap deliveries of 2 orders"""
        new_orders = copy.deepcopy(orders)
        if len(new_orders) < 2:
            return new_orders  # không đủ để swap
        o1, o2 = random.sample(list(new_orders.values()), 2)
        o1.delivery, o2.delivery = o2.delivery, o1.delivery
        return new_orders

    # -------- Main loop --------
    def optimize(self, orders):
        start = time.time()
        best_orders = orders
        best_cost, best_res = self.evaluate(best_orders)
        print(f"Initial objective = {best_cost:.2f}")
        initial_orders=orders
        initial_cost,initial_res = best_cost,best_res
        while time.time() - start < self.max_time:
            improved = False

            for move in [self.couple_exchange, self.block_exchange,
                         self.block_relocate, self.multi_relocate]:
                cand_orders = move(initial_orders)
                cand_cost, cand_res = self.evaluate(cand_orders)
                if cand_cost < best_cost:
                    best_cost, best_res = cand_cost, cand_res
                    best_orders = cand_orders
                    improved = True
                    print(f"Improved to {best_cost:.2f} with {move.__name__}")
                    break
            if not improved:
                # disturbance
                cand_orders = self.disturb(best_orders)
                cand_cost, cand_res = self.evaluate(cand_orders)
                if cand_cost < initial_cost*self.disturb_acceptance:
                    initial_cost, initial_res = cand_cost, cand_res
                    initial_orders = cand_orders
                    print(f"Accepted disturbance -> {initial_cost:.2f}")
                else:
                    break
        return best_cost, best_res
if __name__ == '__main__':
    # (giữ nguyên phần tạo plants, distances, orders, vehicles như bạn viết)
    plants = {
        1: Plant(1, 'P1', docks=2),
        2: Plant(2, 'P2', docks=1),
        3: Plant(3, 'P3', docks=1),
        0: Plant(0, 'Depot', docks=2)
    }
    # distances (symmetric)
    distances = {
        (0, 1): 10.0, (0, 2): 20.0, (0, 3): 25.0,
        (1, 2): 12.0, (1, 3): 15.0, (2, 3): 8.0
    }
    # travel times in minutes (rough)
    travel_times = {k: v / 40.0 * 60.0 for k, v in distances.items()}  # assume avg speed 40 km/h -> minutes

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

    optimizer = VNSMEOptimizer(sim, max_time=5.0)
    best_cost, best_res = optimizer.optimize(orders)

    print("\n=== VNSME Optimization Result ===")
    print("Best objective:", best_cost)
    print("Total late:", best_res["total_late"])
    print("Avg distance:", best_res["avg_distance"])
    for vid,info in best_res["vehicles"].items():
        print("Vehicle", vid, "route:", info["route"])
