import random
import math
import statistics
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import time

class Customer:
    def __init__(self, id, x, y, frequency):
        self.id = id
        self.x = x
        self.y = y
        self.frequency = frequency

    def __repr__(self):
        return f"Customer({self.id}, ({self.x},{self.y}), freq={self.frequency})"

def load_customers(filepath):
    customers = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                id, x, y, freq = map(int, parts)
                customers.append(Customer(id, x, y, freq))
    return customers

def euclidean(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def depot():
    return Customer(0, 0, 0, 0)

def generate_initial_schedule(customers, days=60):
    schedule = defaultdict(list)
    for customer in customers:
        frequency = min(customer.frequency, days)
        chosen_days = random.sample(range(1, days + 1), frequency)
        for day in chosen_days:
            schedule[day].append(customer.id)
    for day in schedule:
        random.shuffle(schedule[day])
    return schedule

def route_distance(route, customer_lookup):
    total = 0
    d = depot()
    if not route:
        return 0
    total += euclidean(d, customer_lookup[route[0]])
    for i in range(len(route) - 1):
        total += euclidean(customer_lookup[route[i]], customer_lookup[route[i + 1]])
    total += euclidean(customer_lookup[route[-1]], d)
    return total

def evaluate_schedule(schedule, customer_lookup):
    total_distance = 0
    day_distances = {}
    for day, route in schedule.items():
        dist = route_distance(route, customer_lookup)
        day_distances[day] = dist
        total_distance += dist
    workload_std = statistics.stdev(day_distances.values()) if len(day_distances) > 1 else 0
    return (round(total_distance, 2), round(workload_std, 2)), day_distances

def mutate_schedule(schedule, days=60):
    new_schedule = copy.deepcopy(schedule)
    mutation_type = random.choice(['day_swap', 'order_shuffle', 'cross_day_swap'])
    if mutation_type == 'day_swap':
        source_day = random.choice(list(new_schedule.keys()))
        if not new_schedule[source_day]: return new_schedule
        customer = random.choice(new_schedule[source_day])
        target_day = random.randint(1, days)
        while target_day == source_day or customer in new_schedule[target_day]:
            target_day = random.randint(1, days)
        new_schedule[source_day].remove(customer)
        new_schedule[target_day].append(customer)
    elif mutation_type == 'order_shuffle':
        target_day = random.choice(list(new_schedule.keys()))
        random.shuffle(new_schedule[target_day])
    elif mutation_type == 'cross_day_swap':
        day1, day2 = random.sample(list(new_schedule.keys()), 2)
        if not new_schedule[day1] or not new_schedule[day2]: return new_schedule
        c1 = random.choice(new_schedule[day1])
        c2 = random.choice(new_schedule[day2])
        if c1 in new_schedule[day2] or c2 in new_schedule[day1]: return new_schedule
        new_schedule[day1].remove(c1)
        new_schedule[day2].remove(c2)
        new_schedule[day1].append(c2)
        new_schedule[day2].append(c1)
    return new_schedule

def crossover_schedules(parent1, parent2, customers, days=60):
    child = defaultdict(list)
    split = random.randint(1, days - 1)
    for day in range(1, split + 1):
        child[day] = parent1.get(day, [])[:]
    for day in range(split + 1, days + 1):
        child[day] = parent2.get(day, [])[:]
    visit_counts = defaultdict(int)
    for day, custs in child.items():
        for c in custs:
            visit_counts[c] += 1
    target_freq = {c.id: min(c.frequency, days) for c in customers}
    for cust_id in target_freq:
        current = visit_counts.get(cust_id, 0)
        diff = target_freq[cust_id] - current
        if diff > 0:
            available_days = [d for d in range(1, days + 1) if cust_id not in child[d]]
            random.shuffle(available_days)
            for i in range(diff):
                if i < len(available_days):
                    child[available_days[i]].append(cust_id)
        elif diff < 0:
            remove_count = -diff
            for d in list(child):
                if cust_id in child[d] and remove_count > 0:
                    child[d].remove(cust_id)
                    remove_count -= 1
    return child

def dominates(f1, f2):
    return (f1[0] <= f2[0] and f1[1] <= f2[1]) and (f1[0] < f2[0] or f1[1] < f2[1])

def compute_crowding(population):
    distances = [f for (_, f) in population]
    n = len(distances)
    crowding = [0.0] * n
    for m in range(2):  # two objectives
        distances_sorted = sorted(enumerate(distances), key=lambda x: x[1][m])
        crowding[distances_sorted[0][0]] = float('inf')
        crowding[distances_sorted[-1][0]] = float('inf')
        f_min = distances_sorted[0][1][m]
        f_max = distances_sorted[-1][1][m]
        for i in range(1, n - 1):
            prev = distances_sorted[i - 1][1][m]
            next = distances_sorted[i + 1][1][m]
            if f_max - f_min == 0:
                norm = 1.0
            else:
                norm = f_max - f_min
            crowding[distances_sorted[i][0]] += (next - prev) / norm
    return crowding

def get_pareto_front(population):
    unique_fitness = {}
    for s, f in population:
        if f not in unique_fitness:
            unique_fitness[f] = s
    filtered = [(s, f) for f, s in unique_fitness.items()]
    pareto = []
    for i, (s1, f1) in enumerate(filtered):
        dominated = False
        for j, (s2, f2) in enumerate(filtered):
            if i != j and dominates(f2, f1):
                dominated = True
                break
        if not dominated:
            pareto.append((s1, f1))
    return pareto

def plot_pareto_front(pareto):
    xs = [f[0] for (_, f) in pareto]
    ys = [f[1] for (_, f) in pareto]
    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, c='blue')
    plt.xlabel("Total Distance")
    plt.ylabel("Workload Imbalance (Std Dev)")
    plt.title("Pareto Front")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pareto_front.png")
    print("Saved Pareto front to pareto_front.png")

if __name__ == '__main__':
    random.seed(42)
    start_time = time.time()

    customers = load_customers('vrp8.txt')
    customer_lookup = {c.id: c for c in customers}
    population_size = 100
    generations = 1000

    population = []
    for _ in range(population_size):
        s = generate_initial_schedule(customers)
        f, _ = evaluate_schedule(s, customer_lookup)
        population.append((s, f))

    best_distances = []
    best_stddevs = []

    for gen in range(generations):
        if (gen + 1) % 10 == 0:
            print(f"[GENERATION {gen + 1}]")
        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(population)[0]
            parent2 = random.choice(population)[0]
            child = crossover_schedules(parent1, parent2, customers)
            child = mutate_schedule(child)
            fitness, _ = evaluate_schedule(child, customer_lookup)
            new_population.append((child, fitness))

        combined = population + new_population
        pareto = get_pareto_front(combined)
        crowding = compute_crowding(pareto)
        # Use dominance-only front for survivor selection
        elite = pareto[:population_size]

        retries = 0
        while len(elite) < population_size and retries < 100:
            s = generate_initial_schedule(customers)
            f, _ = evaluate_schedule(s, customer_lookup)
            elite.append((s, f))
            retries += 1

        population = elite[:population_size]

        # Track best fitness in this generation
        best_fit = min([f for (_, f) in population], key=lambda x: (x[0], x[1]))
        best_distances.append(best_fit[0])
        best_stddevs.append(best_fit[1])

    final_front = get_pareto_front(population)
    plot_pareto_front(final_front)

    # Plot fitness progress with dual Y-axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Total Distance", color='tab:blue')
    ax1.plot(best_distances, color='tab:blue', label='Best Total Distance')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Best Workload Std Dev", color='tab:orange')
    ax2.plot(best_stddevs, color='tab:orange', label='Best Workload Std Dev')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title("Fitness Progress Over Generations")
    plt.savefig("fitness_progress.png")
    print("Saved fitness progress plot to fitness_progress.png")

    # Save fitness to CSV
    with open("fitness_progress.csv", "w") as f:
        f.write("Generation,Best_Distance,Best_StdDev")
        for i in range(generations):
            f.write(f"{i+1},{best_distances[i]},{best_stddevs[i]}")
    print("Saved fitness progress data to fitness_progress.csv")

    # Save Pareto front to CSV for reporting
    with open("pareto_front.csv", "w") as f:
        f.write("Total_Distance,Workload_StdDev")
        for _, (dist, std) in final_front:
            f.write(f"{dist},{std}")
    print("Saved Pareto front to pareto_front.csv")
    print(f"Completed in {time.time() - start_time:.2f} seconds")
