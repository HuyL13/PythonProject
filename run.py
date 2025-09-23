import pickle
from dvrpsim import DVRPSimulator
from dvrp_algorithm import VNSMEOptimizer


for i in range(8):
    print(f"______The {i+1}th order simulation:______ ")
    with open(f"instances/small_50_{i+1}.pkl", "rb") as f:
        data = pickle.load(f)

    plants, orders, vehicles = data["plants"], data["orders"], data["vehicles"]
    distances, travel_times = data["distances"], data["travel_times"]

    sim = DVRPSimulator(plants, orders, vehicles, distances, travel_times,lambda_cost=10000)
    res = sim.simulate_with_tracking(until_time=8*60)
    print(res["objective"])


    optimizer = VNSMEOptimizer(sim, max_time=5.0)
    best_cost, best_res = optimizer.optimize(orders)

    print("\n=== VNSME Optimization Result ===")
    print("Best objective:", best_cost)

