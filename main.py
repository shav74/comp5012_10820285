import random
import math
import statistics
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import time

# creates a customer class to hold customer data
# each customer has an id, x and y coordinates, and a frequency of visits
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

# calculates the Euclidean distance between two customers locations
def euclidean(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# creates a depot customer at the origin (0,0) with frequency 0
# this is the starting point for all routes
def depot():
    return Customer(0, 0, 0, 0)

# generates an initial schedule for the customers as a dictionary
# each customer is assigned to a random day according to their frequency
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

# calculates the total distance of a route
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

# evaluates schedule by calculating the total distance and workload stddev
# the workload is the distance traveled each day
# the standard deviation of the workload is used to measure balance
def evaluate_schedule(schedule, customer_lookup):
    total_distance = 0
    day_distances = {}
    for day, route in schedule.items():
        dist = route_distance(route, customer_lookup)
        day_distances[day] = dist
        total_distance += dist
    workload_std = statistics.stdev(day_distances.values()) if len(day_distances) > 1 else 0
    return (round(total_distance, 2), round(workload_std, 2)), day_distances

# mutates the shedule
# day_swap: swaps a customer from one day to another
# order_shuffle: shuffles the order of customers in a day
# cross_day_swap: swaps customers between two different days
# the mutation type is chosen randomly
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

# crossover schedules by combining two parents to create a child
# the crossover point is chosen randomly
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

# gets the Pareto front from the population
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

# plots the Pareto front and saves it to a file
# the x-axis - total distance and the y-axis - workload stddev
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

# main function to run the genetic algorithm
if __name__ == '__main__':
    random.seed(42)
    start_time = time.time()

    # Load customers from file
    customers = load_customers('data/vrp8.txt')
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

    # Run the genetic algorithm
    print("Starting genetic algorithm...")
    print(f"Population size: {population_size}")
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
    # Identify best solutions from Pareto front
    best_distance = min(final_front, key=lambda x: x[1][0])
    best_balance = min(final_front, key=lambda x: x[1][1])

    # Normalize fitness values to 0–1 for knee point selection
    distances = [f[0] for (_, f) in final_front]
    stddevs = [f[1] for (_, f) in final_front]
    min_dist, max_dist = min(distances), max(distances)
    min_std, max_std = min(stddevs), max(stddevs)

    def normalize(f):
        norm_dist = (f[0] - min_dist) / (max_dist - min_dist) if max_dist != min_dist else 0
        norm_std = (f[1] - min_std) / (max_std - min_std) if max_std != min_std else 0
        return (norm_dist, norm_std)

    knee_point = min(final_front, key=lambda x: math.hypot(*normalize(x[1])))

    print("Best Distance:", best_distance[1])
    print("Best Balance:", best_balance[1])
    print("Knee Point:", knee_point[1])

    # Generate heatmaps for knee, distance, and balance solutions
    import seaborn as sns
    import pandas as pd

    def plot_heatmap(schedule, title, filename):
        heatmap_data = []
        for day in range(1, 61):
            row = [1 if cust in schedule.get(day, []) else 0 for cust in sorted(customer_lookup)]
            heatmap_data.append(row)
        df_heatmap = pd.DataFrame(heatmap_data, columns=[f"C{c}" for c in sorted(customer_lookup)], index=[f"Day {d}" for d in range(1, 61)])
        plt.figure(figsize=(16, 10))
        sns.heatmap(df_heatmap, cmap="YlGnBu", cbar=False)
        plt.title(title)
        plt.xlabel("Customers")
        plt.ylabel("Day")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved heatmap to {filename}")

    # Plot bar chart of route distance per day for each key solution
    def plot_daily_distance_bar(schedule, label, filename):
        daily_dists = []
        for day in range(1, 61):
            route = schedule.get(day, [])
            dist = route_distance(route, customer_lookup) if route else 0
            daily_dists.append(dist)
        plt.figure(figsize=(12, 5))
        plt.bar(range(1, 61), daily_dists, color='steelblue')
        plt.xlabel("Day")
        plt.ylabel("Route Distance")
        plt.title(f"Daily Route Distances – {label}")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved bar chart to {filename}")

    plot_daily_distance_bar(knee_point[0], "Knee Point", "knee_point_barchart.png")
    plot_daily_distance_bar(best_distance[0], "Best Distance", "best_distance_barchart.png")
    plot_daily_distance_bar(best_balance[0], "Best Balance", "best_balance_barchart.png")

    plot_heatmap(knee_point[0], "Knee Point – Customer Visits per Day", "knee_point_heatmap.png")
    plot_heatmap(best_distance[0], "Best Distance – Customer Visits per Day", "best_distance_heatmap.png")
    plot_heatmap(best_balance[0], "Best Balance – Customer Visits per Day", "best_balance_heatmap.png")
    print("Saved heatmap for knee point to knee_point_heatmap.png")

    plot_pareto_front(final_front)

    # Plot fitness progress
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Total Distance", color='tab:blue')
    ax1.plot(best_distances, color='tab:blue', label='Best Total Distance')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for the standard deviation
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
