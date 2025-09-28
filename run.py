from simulation.helpers import *
from distributions import sample_truncated_normal

# Parameters
SIMULATIONS = 1000
RUN_UNTIL = 100

num_queues = [1, 3, 5, 7, 9, 11, 13]
avg_waiting_times = []
arrival_rate = 10 # minutes
service_rate = 1/2 # minutes

# Defining the truncated normal distribution for service times
mu = service_rate
sigma = 1/6  # a sixth of a minute (10 seconds)

mu_additional = 2
sigma_additional = 2  

service_distribution = (0.5, 1/6)
additional_screening_distribution = (2.0, 2.0)

# Run simulations for different queue numbers
print("\n--- Multiple Queue Airport Simulation ---\n")
for n in num_queues:
    
    print(f"\nRunning simulation with {n} queue(s)...")
    
    queue_waiting_times = []  # Store waiting times for this queue configuration

    for _ in tqdm(range(SIMULATIONS), desc="Simulations"):
        avg_queue_lens, arrivals_multi, all_passengers, max_queue_length = run_simulation(arrival_rate=arrival_rate, service_rate=service_rate, num_queues=n, service_distribution=service_distribution, additional_screening_distribution=additional_screening_distribution, run_until=RUN_UNTIL)

        waiting_times = []
        for passenger in all_passengers.values():
            if passenger.service_start_time is not None and passenger.service_end_time is not None:
                waiting_times.append(passenger.total_queue_time())   # waiting only

        # Calculate average waiting time for this simulation
        avg_wait_time = np.mean(waiting_times) 
        queue_waiting_times.append(avg_wait_time)

    # Calculate overall average for this queue configuration across all simulations
    overall_avg = np.mean(queue_waiting_times)
    avg_waiting_times.append(overall_avg)

    print(f"Average waiting time (simulations = {SIMULATIONS}, run_until = {RUN_UNTIL}): {overall_avg:.3f} minutes")

# Plot average waiting time vs number of queues and avg queue length
plot_waiting_time_vs_queues(num_queues, avg_waiting_times)
plot_queue_length(avg_queue_lens, arrivals_multi)