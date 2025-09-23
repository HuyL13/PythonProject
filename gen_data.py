import os
import pickle
import random
from dvrpsim import DVRPSimulator,Vehicle, Plant,Order
# Hàm generate_instance từ trước giữ nguyên
def generate_instance(n_plants: int, n_orders: int, n_vehicles: int,
                      capacity_range=(3,10), docks_range=(1,3), seed=None):
    if seed is not None:
        random.seed(seed)

    plants = {pid: Plant(id=pid, name=f"P{pid}", docks=random.randint(*docks_range))
              for pid in range(n_plants)}

    distances = {}
    for i in range(n_plants):
        for j in range(i+1, n_plants):
            d = round(random.uniform(5.0, 50.0), 1)
            distances[(i, j)] = d
            distances[(j, i)] = d
    travel_times = {k: v/40.0*60.0 for k, v in distances.items()}

    orders = {}
    for oid in range(1, n_orders+1):
        pickup = random.randint(0, n_plants-1)
        delivery = random.randint(0, n_plants-1)
        while delivery == pickup:
            delivery = random.randint(0, n_plants-1)
        qty = random.randint(1, 5)
        arrival = random.randint(0, 300)
        due = arrival + random.randint(60, 300)
        orders[oid] = Order(id=oid, pickup=pickup, delivery=delivery,
                            qty=qty, arrival=arrival, due=due)

    vehicles = {vid: Vehicle(id=vid,
                             capacity=random.randint(*capacity_range),
                             start_location=0)
                for vid in range(1, n_vehicles+1)}

    return plants, orders, vehicles, distances, travel_times


def generate_and_save_all(out_dir="instances"):
    os.makedirs(out_dir, exist_ok=True)

    configs = []
    # small: 50, 100, 200 orders
    for orders in [50, 100, 200]:
        for _ in range(8):  # 8 instance cho mỗi mức
            configs.append(("small", orders))
    # medium: 500, 1000 orders
    for orders in [500, 1000]:
        for _ in range(8):  # 8 instance cho mỗi mức
            configs.append(("medium", orders))
    # big: 2000, 4000 orders
    for orders in [2000, 4000]:
        for _ in range(8):  # 8 instance cho mỗi mức
            configs.append(("big", orders))

    print(f"Tổng số instance: {len(configs)}")  # = 64

    for idx, (size, orders) in enumerate(configs, 1):
        plants, o, v, dist, ttime = generate_instance(
            n_plants=154,
            n_orders=orders,
            n_vehicles=random.randint(5, 100),
            seed=idx  # để reproducible
        )
        data = dict(plants=plants, orders=o, vehicles=v,
                    distances=dist, travel_times=ttime)

        fname = os.path.join(out_dir, f"{size}_{orders}_{idx}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved {fname}")


if __name__ == "__main__":
    generate_and_save_all()
