import heapq
import random
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from tqdm import tqdm

from distributions import sample_truncated_normal

class Passenger:
    def __init__(self, id, arrival_time, service_start_time=None, service_end_time=None):
        self.id = id
        self.arrival_time = arrival_time
        self.service_start_time = service_start_time
        self.service_end_time = service_end_time
        # Determine if the passenger needs additional screening using a Bernoulli trial with 3% prob
        self.needs_additional_screening = True if sts.bernoulli.rvs(p=0.03) == 1 else False 

    def total_queue_time(self):
        return self.service_start_time - self.arrival_time

    def total_service_time(self):
        return self.service_end_time - self.service_start_time

    def __repr__(self):
        waiting_time = f"{self.total_queue_time():.2f}" if self.total_queue_time() is not None else "N/A"
        service_time = f"{self.total_service_time():.2f}" if self.total_service_time() is not None else "N/A"
        
        return (f"Passenger {self.id}: Arrival Time = {self.arrival_time:.2f},"
                f"Total Waiting Time = {waiting_time},"
                f"Total Service Time = {service_time},"
                f"Additional Screening = {self.needs_additional_screening}")

class Queue:
    def __init__(self, timestamp, service_rate, service_distribution=None, additional_screening_distribution=None):
        self.timestamp = timestamp
        self.service_rate = service_rate
        self.waiting = []
        self.being_served = None  # Single passenger being served (or None if server is free)
        
        if service_distribution is None:
            mu = service_rate
            sigma = 1/6
            service_distribution = sample_truncated_normal(mu, sigma)
            
        self.service_distribution = service_distribution
        
        if additional_screening_distribution is None:
            additional_mu = 2  # minutes
            additional_sigma = 2  # minutes
            additional_screening_distribution = sample_truncated_normal(additional_mu, additional_sigma)

        self.additional_screening_distribution = additional_screening_distribution
        
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    # Calculate the duration of the service for a particular person in the queue
    def determine_service_duration(self, passenger):
        # Regular service duration
        base_service_duration = self.service_distribution
        
        if passenger.needs_additional_screening:
            additional_service_duration = self.additional_screening_distribution
            total_Service_duration = base_service_duration + additional_service_duration
        else:
            total_Service_duration = base_service_duration

        return total_Service_duration

    def start_service(self, passenger, now):
        self.being_served = passenger
        passenger.service_start_time = now
        service_duration = self.determine_service_duration(passenger)
        end_service_time = now + service_duration
        passenger.service_end_time = end_service_time

        return end_service_time
        
    def end_service(self):
        self.being_served = None

    def add_passenger(self, passenger):
        self.waiting.append(passenger)
    
    def get_next_passenger(self):
        if self.waiting:
            return self.waiting.pop(0)
        return None
    
    def waiting_count(self):
        return len(self.waiting)
    
    def being_served_count(self):
        return 1 if self.being_served else 0
    
    def total_passengers(self):
        return self.waiting_count() + self.being_served_count()
            
class Airport:
    # Sample the first arrival from the exponential distribution
    # Here, we have two queues, one for the passengers created by the Queue() class, and another priority queue that schedules events. The reason for that is that we need one structure that triggers the service, and another one that schedules when the passaengers will arrive. 
    def __init__(self, arrival_rate, service_rate, num_queues=1, service_distribution=None, additional_screening_distribution=None):
        self.priority_q = []
        self.now = 0
        self.arrival_rate = arrival_rate
        self.num_queues = num_queues
        self.passenger_queues = [Queue(timestamp=0, service_rate=service_rate, service_distribution=service_distribution, additional_screening_distribution=additional_screening_distribution) for _ in range(num_queues)] # multiple queues
        self.curr_arrival_time = 0
        self.passenger_counter = 0  # for unique passenger IDs
        self.all_passengers = {}  # store all passengers for analysis
    
    def find_shortest_queue(self):
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
        self.all_passengers[passenger.id] = passenger 
        selected_queue.add_passenger(passenger)
        
        # Create passenger and schedule arrival
        heapq.heappush(self.priority_q, (arrival_time, (1, selected_queue, passenger.id)))
    
    def run_next_service(self):
        timestamp, event_info = heapq.heappop(self.priority_q)
        self.now = timestamp

        event_type, queue, passenger_id = event_info  # now both events have the same structure

        if event_type == 1:  # arrival
            if queue.being_served_count() == 0:  # if the server is free
                passenger = queue.get_next_passenger()  # remove them from waiting
                end_service_time = queue.start_service(passenger, self.now)
                heapq.heappush(self.priority_q, (end_service_time, (0, queue, passenger_id)))  # schedule departure
            else:
                pass  # server is busy, do nothing (since the passenger is already in waiting queue)

        elif event_type == 0:  # departure
            queue.end_service()
            passenger = queue.get_next_passenger()

            if passenger: # if there's people still waiting in this specific queue
                end_service_time = queue.start_service(passenger, self.now)
                heapq.heappush(self.priority_q, (end_service_time, (0, queue, passenger.id))) # schedule departure
                               
    def __repr__(self):
        total_waiting = sum(q.waiting_count() for q in self.passenger_queues)
        total_being_served = sum(q.being_served_count() for q in self.passenger_queues)
        
        result = f"Airport at time (minutes) {np.round(self.now, 3)}: {total_waiting} total waiting, {total_being_served} total being served"
        
        for i, q in enumerate(self.passenger_queues):
            result += f"\n  Queue {i+1}: {q.waiting_count()} waiting, {q.being_served_count()} being served."

        additional_screening_count = sum(1 for p in self.all_passengers.values() if p.needs_additional_screening)
        result += f"\n  Additional Screening: {additional_screening_count} needed it so far."
        return result
    
    def print_schedule(self):
        print(repr(self))
        for time, event in sorted(self.priority_q):
            event_type, queue, passenger_id = event
            # Check if a passenger needed additional screening
            needed_screening = True if self.all_passengers[passenger_id].needs_additional_screening else False
            queue_index = self.passenger_queues.index(queue)
            event_name = "arrival" if event_type == 1 else "departure"
            
            if needed_screening:
                print(f"\t [Additional Screening] Scheduled Timestamp (minutes) {np.round(time, 2)}: {event_name} for passenger {passenger_id} at queue {queue_index + 1}")
            else:
                print(f"\t Scheduled Timestamp (minutes) {np.round(time, 2)}: {event_name} for passenger {passenger_id} at queue {queue_index + 1}")

        print("\n")