## This code was inspired by the code provided in the screencast: https://nbviewer.jupyter.org/urls/course-resources.minerva.edu/uploaded_files/mke/00229796-8438/simulations-with-schedules.ipynb

import heapq
import random
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from tqdm import tqdm

# I want to minimize the average total time a passenger spends in line in the system (waiting + being served). What's the amount of queues I should have to achieve that?
class Passenger:
    def __init__(self, id, arrival_time, service_start_time=None, service_end_time=None):
        self.id = id
        self.arrival_time = arrival_time
        self.service_start_time = service_start_time
        self.service_end_time = service_end_time
        self.needs_additional_screening = False

    def total_queue_time(self):
        return self.service_start_time - self.arrival_time

    def total_service_time(self):
        return self.service_end_time - self.service_start_time

    def __repr__(self):
        service_start = f"{self.service_start_time:.2f}" if self.service_start_time is not None else "N/A"
        service_end = f"{self.service_end_time:.2f}" if self.service_end_time is not None else "N/A"
        waiting_time = f"{self.total_queue_time():.2f}" if self.total_queue_time() is not None else "N/A"
        service_time = f"{self.total_service_time():.2f}" if self.total_service_time() is not None else "N/A"
        return (f"Passenger {self.id}: Arrival Time = {self.arrival_time:.2f}, "
                f"Service Start Time = {service_start}, "
                f"Service End Time = {service_end}, "
                f"Total Waiting Time = {waiting_time}, "
                f"Total Service Time = {service_time}")

class Queue:
    """
    A queue representing a single-server system

    Attributes:
        timestamp : float
            The time associated with the queue (used for scheduling comparisons).
        service_rate : float
            Average service rate (mean duration of service).
        waiting_passengers : list
            List of passengers currently waiting in the queue.
        being_served_passenger : Passenger or None
            The passenger currently being served (None if server is free).
    """
    def __init__(self, timestamp, service_rate):
        self.timestamp = timestamp
        self.service_rate = service_rate
        self.waiting = []
        self.being_served = None  # Single passenger being served (or None if server is free)
        
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    def needs_additional_screening(self):
        return sts.bernoulli.rvs(p=0.03)
    
    # Calculate the duration of the service for a particular person in the queue
    def determine_service_duration(self, passenger):
        def sample_truncated_normal(mu, sigma, lower_bound=0, upper_bound=np.inf):
            a, b = abs((lower_bound - mu)/sigma), (upper_bound - mu)/sigma
            distribution = sts.truncnorm(a, b)
            return distribution.rvs(size=1)
        
        # Regular service duration
        mu = self.service_rate  # minutes
        sigma = 1/6  # a sixth of a minute 
        base_service_duration = sample_truncated_normal(mu, sigma)
        
        # Check if additional screening is needed
        if self.needs_additional_screening():
            passenger.needs_additional_screening = True
            # Add additional screening time from truncated normal distribution
            additional_mu = 2  # minutes
            additional_sigma = 2  # minutes
            additional_screening_duration = sample_truncated_normal(additional_mu, additional_sigma)
            total_service_duration = base_service_duration + additional_screening_duration
        else:
            total_service_duration = base_service_duration
        
        return total_service_duration
    
    def start_service(self, passenger, now):

        self.being_served = passenger
        passenger.service_start_time = now
        service_duration = self.determine_service_duration(passenger)
        end_service_time = now + service_duration[0]  # Extract from array
        passenger.service_end_time = end_service_time

        return end_service_time
        
    def end_service(self):
        """Mark the end of service for the current passenger"""
        self.being_served = None

    def add_passenger(self, passenger):
        """Add a passenger to the waiting queue."""
        self.waiting.append(passenger)
    
    def get_next_passenger(self):
        """Get the next passenger from the waiting queue."""
        if self.waiting:
            return self.waiting.pop(0)
        return None
    
    def waiting_count(self):
        """Number of passengers waiting."""
        return len(self.waiting)
    
    def being_served_count(self):
        """Number of passengers being served (0 or 1)."""
        return 1 if self.being_served else 0
    
    def total_passengers(self):
        """Total passengers in queue (waiting + being served)."""
        return self.waiting_count() + self.being_served_count()
            
class Airport:
    """
    Simulates an airport passenger arrival and service process.

    Attributes:
        priority_q : list of tuple
            Priority queue of scheduled events (time, event_type).
            Event types: 1 = arrival, 0 = departure.
        now : float
            Current simulation time.
        arrival_rate : float
            Average arrival rate (passengers per minute).
        passenger_q : Queue
            Queue object managing waiting and serving passengers.
        curr_arrival_time : float
            Timestamp of the most recently scheduled arrival.
    """
    
    # Sample the first arrival from the exponential distribution
    # Here, we have two queues, one for the passengers created by the Queue() class, and another priority queue that schedules events. The reason for that is that we need one structure that triggers the service, and another one that schedules when the passaengers will arrive. 
    def __init__(self, arrival_rate, service_rate, num_queues=1):
        """
        Initialize airport simulation.

        Parameters:
            arrival_rate : float
                Average arrival rate (passengers per minute).
            service_rate : float
                Average service rate (mean service duration in minutes).
            num_queues : int
                Number of parallel queues/service stations (default: 1).
        """
        self.priority_q = []
        self.now = 0
        self.arrival_rate = arrival_rate
        self.num_queues = num_queues
        self.passenger_queues = [Queue(timestamp=0, service_rate=service_rate) for _ in range(num_queues)] # multiple queues
        self.curr_arrival_time = 0
        self.passenger_counter = 0  # for unique passenger IDs
        self.completed_passengers = []  # store completed passengers for analysis
    
    def find_shortest_queue(self):
        """
        Find the queue with the shortest total length (waiting + being served).
        
        Returns:
            Queue: The queue with the minimum total passengers.
        """
        # if all queues have the same length, sample one randomly
        first_q = self.passenger_queues[0].total_passengers()
        if all(q.total_passengers() == first_q for q in self.passenger_queues):
            queue_idx = random.randint(0, self.num_queues - 1) # get the index of the queue
            return self.passenger_queues[queue_idx]

        return min(self.passenger_queues, key=lambda q: q.total_passengers())
                   
    def add_arrival(self, now):
        """
        Schedule the next passenger arrival.

        now : float
            Current simulation time.

        Event encoding:
          1 = passenger arrival
          0 = passenger departure
        """
        arrival_time = now
        distribution = sts.expon(scale=1/self.arrival_rate)
        sampled_interarrival_time = distribution.rvs(size=1)
        arrival_time += sampled_interarrival_time[0] 
        self.curr_arrival_time = arrival_time
        
        # Select shortest queue to add the passenger to
        selected_queue = self.find_shortest_queue()
        
        self.passenger_counter += 1
        passenger = Passenger(id=self.passenger_counter, arrival_time=arrival_time)
        selected_queue.add_passenger(passenger)
        
        # Create passenger and schedule arrival
        heapq.heappush(self.priority_q, (arrival_time, (1, selected_queue, passenger.id)))
    
    def run_next_service(self):
        """
        Execute the next scheduled event (arrival or departure).
        - If a passenger arrives, find the shortest queue and either start service or add to waiting.
        - If a passenger departs, free the server and check for waiting passengers in that queue.
        """
        timestamp, event_info = heapq.heappop(self.priority_q)
        self.now = timestamp

        event_type, queue, passenger_id = event_info  # now both events have the same structure

        if event_type == 1:  # arrival
            if queue.being_served_count() == 0:  # if the server is free
                passenger = queue.get_next_passenger()  # remove them from waiting
                end_service_time = queue.start_service(passenger, self.now)
                heapq.heappush(self.priority_q, (end_service_time, (0, queue, passenger.id)))  # schedule departure
            else:
                pass  # server is busy, do nothing (since the passenger is already in waiting queue)

        elif event_type == 0:  # departure
            if queue.being_served: # if there's someone in that queue being served
                self.completed_passengers.append(queue.being_served)
            
            queue.end_service()
            passenger = queue.get_next_passenger()

            if passenger: # if there's people still waiting in this specific queue
                end_service_time = queue.start_service(passenger, self.now)
                heapq.heappush(self.priority_q, (end_service_time, (0, queue, passenger.id))) # schedule departure
                               
    def __repr__(self):
        """
        Represent the current state of the simulation.

        Returns:
        str
            Current time and status of all queues.
        """
        total_waiting = sum(q.waiting_count() for q in self.passenger_queues)
        total_being_served = sum(q.being_served_count() for q in self.passenger_queues)
        
        result = f"Airport at time (minutes) {np.round(self.now, 3)}: {total_waiting} total waiting, {total_being_served} total being served"
        
        # Show individual queue details
        additional_screening_count = sum(1 for passenger in self.completed_passengers if passenger.needs_additional_screening) # also display how many passengers needed additional screening
        for i, q in enumerate(self.passenger_queues):
            result += f"\n  Queue {i+1}: {q.waiting_count()} waiting, {q.being_served_count()} being served."
            result += f" {additional_screening_count} needed additional screening so far."
        return result
    
    def print_schedule(self):
        print(repr(self))
        for time, event in sorted(self.priority_q):
            event_type, queue, passenger_id = event
            queue_index = self.passenger_queues.index(queue)
            event_name = "arrival" if event_type == 1 else "departure"
            print(f"\t Scheduled Timestamp (minutes) {np.round(time, 2)}: {event_name} for passenger {passenger_id} at queue {queue_index}")
        print("\n")

## Create run_simulation
def run_simulation(arrival_rate, service_rate, run_until, num_queues=1):
    """
    Run a simulation of an airport queue for a discrete number of times

    Parameters:
        arrival_rate : float
            Average passenger arrival rate (passengers per minute).
        service_rate : float
            Average service rate (mean service duration in minutes).
        run_until : int
            Number of events (arrivals) to simulate.
        num_queues : int
            Number of parallel queues/service stations (default: 1).

    Returns:
        tuple of (list, list, list)
            queue_over_time : list of int
                Total number of passengers waiting in all queues at each event.
            arrival_times : list of float
                Scheduled arrival times for each passenger.
            completed_passengers : list of Passenger
                List of passengers who completed service.
    """
    queue_over_time = []
    arrival_times = []
    airport = Airport(arrival_rate, service_rate, num_queues)
    
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
        
        # Record queue state
        total_waiting = sum(q.waiting_count() for q in airport.passenger_queues)
        queue_over_time.append(total_waiting)
        arrival_times.append(airport.now)
    
    return queue_over_time, arrival_times, airport.completed_passengers

def plot_waiting_time_vs_queues(queue_numbers, avg_waiting_times):
    """
    Plot average waiting time versus number of queues.
    
    Parameters:
        queue_numbers : list of int
            List of queue numbers tested.
        avg_waiting_times : list of float
            Corresponding average waiting times.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(queue_numbers, avg_waiting_times, 'bo-', linewidth=2, markersize=8)
    plt.title('Average Waiting Time vs Number of Queues', fontsize=14)
    plt.xlabel('Number of Queues', fontsize=12)
    plt.ylabel('Average Waiting Time (minutes)', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print(f"\nAverage Waiting Time Results\n")
    for queues, time in zip(queue_numbers, avg_waiting_times):
        print(f"{queues} queue(s): {time:.2f} minutes")
            

def plot_queue_length(queue, arrivals):
    """
    Plot the simulated queue length over time.

    Parameters:
        queue : list of int
            Queue length recorded at each simulated event.
        arrivals : list of float
            Arrival times corresponding to each queue observation.
    """
    plt.figure()
    plt.title('Simulated queue length over time')
    plt.plot(arrivals, queue, 'k.')
    plt.xlabel('Arrival Times [min]')
    plt.ylabel('Queue Length')
    plt.show()
    
    # print average queue length
    avg_queue_length = np.mean(queue)
    print(f"Average queue length: {avg_queue_length:.2f}")
       
# Example usage:
if __name__ == "__main__":
    print("\n--- Multiple Queue Simulation ---\n")
    
    # Parameters
    simulations = 100
    run_until = 20
    queue_lengths = [1, 3, 5, 7, 9, 11, 13]
    avg_waiting_times = []
    arrival_rate = 10 # minutes
    service_rate = 1/2 # minutes
    
    # Run simulations for different queue numbers
    for num_queues in queue_lengths:
        print(f"\nRunning simulation with {num_queues} queue(s)...")
        
        queue_waiting_times = []  # Store waiting times for this queue configuration

        for _ in tqdm(range(simulations), desc="Simulations"):
            queue_multi, arrivals_multi, completed_passengers = run_simulation(arrival_rate=arrival_rate, service_rate=service_rate, run_until=run_until, num_queues=num_queues)
            
            waiting_times = []
            for passenger in completed_passengers:
                total_waiting_time = passenger.total_queue_time() + passenger.total_service_time()
                waiting_times.append(total_waiting_time)

            # Calculate average waiting time for this simulation
            avg_wait_time = np.mean(waiting_times) 
            queue_waiting_times.append(avg_wait_time)
        
        # Calculate overall average for this queue configuration
        overall_avg = np.mean(queue_waiting_times)
        avg_waiting_times.append(overall_avg)
        
        print(f"Average waiting time (1000 simulations run_until = 20): {overall_avg:.3f} minutes")
    
    # Plot average waiting time vs number of queues
    plot_waiting_time_vs_queues(queue_lengths, avg_waiting_times)