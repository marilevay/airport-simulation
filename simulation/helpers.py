from simulation.classes import Passenger, Queue, Airport
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from tqdm import tqdm

## Create run_simulation
def run_simulation(arrival_rate, service_rate, run_until, num_queues=1, service_distribution=None, additional_screening_distribution=None):
    avg_queue_lens = []
    arrival_times = []
    max_queue_length = 0
    airport = Airport(arrival_rate, service_rate, num_queues=num_queues, service_distribution=service_distribution, additional_screening_distribution=additional_screening_distribution)
    
    # Schedule the specified number of arrivals
    current_time = 0
    for _ in range(run_until):
        airport.add_arrival(current_time)
        current_time = airport.curr_arrival_time
    
    # Now process all events (arrivals and departures) until priority queue is empty
    while airport.priority_q:  # Safety limit to prevent infinite loops
        airport.run_next_service()
            
        # record average queue state (waiting + in service)
        avg_state = np.mean([q.total_passengers() for q in airport.passenger_queues])
        avg_queue_lens.append(avg_state)
        arrival_times.append(airport.now)
            
        # track the maximum of any single queue (waiting + in service)
        current_max = max(q.total_passengers() for q in airport.passenger_queues)
        if current_max > max_queue_length:
            max_queue_length = current_max

    
    return avg_queue_lens, arrival_times, airport.all_passengers, max_queue_length

def plot_waiting_time_vs_queues(number_of_queues, avg_waiting_times):
    plt.figure(figsize=(10, 6))
    plt.plot(number_of_queues, avg_waiting_times, 'bo-', linewidth=2, markersize=8)
    plt.title('Average Waiting Time vs Number of Queues', fontsize=14)
    plt.xlabel('Number of Queues', fontsize=12)
    plt.ylabel('Average Waiting Time (minutes)', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print(f"\nAverage Waiting Time Results\n")
    for queues, time in zip(number_of_queues, avg_waiting_times):
        print(f"{queues} queue(s): {time:.2f} minutes")
            

def plot_queue_length(avg_queue_lens, arrivals):
    plt.figure()
    plt.title('Average Queue Length Over Time')
    plt.plot(arrivals, avg_queue_lens, 'k.')
    plt.xlabel('Arrival Times [min]')
    plt.ylabel('Average Queue Length (all queues)')
    plt.show()
    
    # print average queue length across all queues and all times
    avg_queue_length = np.mean(avg_queue_lens)
    print(f"\nAverage queue length (across all queues and simulations): {avg_queue_length:.2f}\n")
    
def collect_metrics_once(arrival_rate, service_rate, run_until, num_queues=1,
                         service_distribution=None, additional_screening_distribution=None):
    """
    Run a single simulation and collect performance metrics.

    Returns
    -------
    dict with keys:
        - avg_waiting_time
        - avg_service_time
        - avg_total_time
        - avg_queue_length
        - max_queue_length
        - num_passengers
    """
    avg_queue_lens, arrival_times, passengers, max_queue_length = run_simulation(
        arrival_rate, service_rate, run_until,
        num_queues=num_queues,
        service_distribution=service_distribution,
        additional_screening_distribution=additional_screening_distribution
    )

    waiting_times = []
    service_times = []
    total_times = []

    for p in passengers.values():
        if p.service_start_time is not None and p.service_end_time is not None:
            wait = p.total_queue_time()
            service = p.total_service_time()
            total = wait + service
            waiting_times.append(wait)
            service_times.append(service)
            total_times.append(total)

    metrics = {
        "avg_waiting_time": np.mean(waiting_times) if waiting_times else 0,
        "avg_service_time": np.mean(service_times) if service_times else 0,
        "avg_total_time": np.mean(total_times) if total_times else 0,
        "avg_queue_length": np.mean(avg_queue_lens) if avg_queue_lens else 0,
        "max_queue_length": max_queue_length,
        "num_passengers": len(passengers)
    }

    return metrics

def collect_metrics_many(arrival_rate, service_rate, run_until, num_queues=1,
                         service_distribution=None, additional_screening_distribution=None,
                         num_replications=100):
    """
    Run multiple simulations and compute mean + 95% CI for each metric.

    Returns
    -------
    dict with structure:
        {
            "avg_waiting_time": (mean, ci_low, ci_high),
            "avg_service_time": (mean, ci_low, ci_high),
            "avg_total_time": (mean, ci_low, ci_high),
            "avg_queue_length": (mean, ci_low, ci_high),
            "max_queue_length": (mean, ci_low, ci_high),
            "num_passengers": (mean, ci_low, ci_high)
        }
    """
    all_metrics = {k: [] for k in [
        "avg_waiting_time", "avg_service_time", "avg_total_time",
        "avg_queue_length", "max_queue_length", "num_passengers"
    ]}

    for _ in range(num_replications):
        metrics = collect_metrics_once(
            arrival_rate, service_rate, run_until, num_queues,
            service_distribution, additional_screening_distribution
        )
        for k, v in metrics.items():
            all_metrics[k].append(v)

    results = {}
    for k, values in all_metrics.items():
        mean = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        z = 1.96  # 95% CI
        ci_low, ci_high = mean - z * se, mean + z * se
        results[k] = (mean, ci_low, ci_high)

    return results
