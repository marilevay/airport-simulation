from simulation.classes import Passenger, Queue, Airport
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from tqdm import tqdm

## Create run_simulation
def run_simulation(arrival_rate, service_rate, run_until, num_queues=1, service_distribution=None, additional_screening_distribution=None):
    avg_queue_lens = []
    arrival_times = []
    airport = Airport(arrival_rate, service_rate, num_queues=num_queues, service_distribution=service_distribution, additional_screening_distribution=additional_screening_distribution)
    
    # Schedule the specified number of arrivals
    current_time = 0
    for _ in range(run_until):
        airport.add_arrival(current_time)
        current_time = airport.curr_arrival_time
    
    # Now process all events (arrivals and departures) until priority queue is empty
    event_count = 0
    while airport.priority_q:  # Safety limit to prevent infinite loops
        airport.run_next_service()
        event_count += 1
        
        # record average queue state
        avg_waiting = np.mean([q.waiting_count() for q in airport.passenger_queues])
        avg_queue_lens.append(avg_waiting)
        arrival_times.append(airport.now)
    
    return avg_queue_lens, arrival_times, airport.all_passengers

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