import heapq
import random
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from tqdm import tqdm

from distributions import sample_truncated_normal

class Passenger:
    def __init__(self, id, arrival_time, service_start_time=None, service_end_time=None, p_screen=0.03):
        """Initialize a passenger with arrival and service times."""
        self.id = id
        self.arrival_time = arrival_time
        self.service_start_time = service_start_time
        self.service_end_time = service_end_time
        self.base_service_end_time = None  # optional, included here for analysis
        self.needs_additional_screening = bool(sts.bernoulli.rvs(p=p_screen))

    def total_queue_time(self):
        """Return time spent waiting in queue."""
        if self.service_start_time is None:
            return None
        return max(0.0, self.service_start_time - self.arrival_time)

    def total_service_time(self):
        """Return time spent being served."""
        if self.service_start_time is None or self.service_end_time is None:
            return None
        return max(0.0, self.service_end_time - self.service_start_time)


    def __repr__(self):
        """String summary of passenger's times and screening status."""
        waiting_time = f"{self.total_queue_time():.2f}" if self.total_queue_time() is not None else "N/A"
        service_time = f"{self.total_service_time():.2f}" if self.total_service_time() is not None else "N/A"
        
        return (f"Passenger {self.id}: Arrival Time = {self.arrival_time:.2f},"
                f"Total Waiting Time = {waiting_time},"
                f"Total Service Time = {service_time},"
                f"Additional Screening = {self.needs_additional_screening}")

class Queue:
    def __init__(self, timestamp, service_rate, service_distribution=None, additional_screening_distribution=None):
        """Queue for passengers, tracks service and waiting."""
        self.timestamp = timestamp
        self.service_rate = service_rate
        self.waiting = []
        self.being_served = None  # Single passenger being served (or None if server is free)
        
        if service_distribution is None:
            self.service_mu, self.service_sigma = 0.5, 1/6
        else:
            self.service_mu, self.service_sigma = service_distribution

        if additional_screening_distribution is None:
            self.add_mu, self.add_sigma = 2.0, 2.0  # mean of 2 minutes, stddev of 2 minutes
        else:
            self.add_mu, self.add_sigma = additional_screening_distribution
        
    def __lt__(self, other):
        """Compare queues by timestamp for priority queue."""
        return self.timestamp < other.timestamp
    
    # Calculate the duration of the service for a particular person in the queue
    def determine_service_duration(self, passenger):
        """Sample base service duration for a passenger."""
        # Only return the base service at the checkpoint
        return sample_truncated_normal(self.service_mu, self.service_sigma)

    def start_service(self, passenger, now):
        """
        Begin serving a passenger
        
        Inputs:
            passenger (Passenger): The passenger to serve
            now (float): Current simulation time
            
        Outputs:
            float: Time when service ends for this passenger
        """
        self.being_served = passenger
        passenger.service_start_time = now
        service_duration = self.determine_service_duration(passenger)
        end_service_time = now + service_duration
        passenger.base_service_end_time = end_service_time  # checkpoint done

        return end_service_time
        
    def end_service(self):
        """Mark server as free"""
        self.being_served = None

    def add_passenger(self, passenger):
        """Add a passenger to the waiting list"""
        self.waiting.append(passenger)
    
    def get_next_passenger(self):
        """
        Pop and return the next passenger from waiting list
    
        Outputs:
            Passenger or None: The next passenger, or None if waiting list is empty
        """
        if self.waiting:
            return self.waiting.pop(0)
        return None
    
    def waiting_count(self):
        """Return number of waiting passengers"""
        return len(self.waiting)
    
    def being_served_count(self):
        """Return 1 if serving, else 0."""
        return 1 if self.being_served else 0
    
    def total_passengers(self):
        """Return total passengers (waiting + served)"""
        return self.waiting_count() + self.being_served_count()
            
class Airport:
    # Sample the first arrival from the exponential distribution
    # Here, we have two queues, one for the passengers created by the Queue() class, and another priority queue that schedules events. The reason for that is that we need one structure that triggers the service, and another one that schedules when the passaengers will arrive. 
    def __init__(self, arrival_rate, service_rate, num_queues=1, service_distribution=None, additional_screening_distribution=None):
        """Airport simulation with queues and event scheduling."""
        self.priority_q = []
        self.now = 0
        self.officer_busy_until = 0
        self.arrival_rate = arrival_rate
        self.num_queues = num_queues
        self.screening_prob = 0.03
        self.passenger_queues = [Queue(timestamp=0, service_rate=service_rate, service_distribution=service_distribution, additional_screening_distribution=additional_screening_distribution) for _ in range(num_queues)] # multiple queues
        self.curr_arrival_time = 0
        self.passenger_counter = 0  # for unique passenger IDs
        self.all_passengers = {}  # store all passengers for analysis
    
    def find_shortest_queue(self):
        """
        Find the queue with the fewest passengers.
        
        Outputs:
            Queue: The selected queue (random if tied).
        """
        # if all queues have the same length, sample one randomly
        first_q = self.passenger_queues[0].total_passengers()
        if all(q.total_passengers() == first_q for q in self.passenger_queues):
            queue_idx = random.randint(0, self.num_queues - 1) # get the index of the queue
            return self.passenger_queues[queue_idx]

        return min(self.passenger_queues, key=lambda q: q.total_passengers())
                   
    def add_arrival(self, now):
        """
        Schedule the next passenger arrival event.
        
        Inputs:
            now (float): Current simulation time.
            
        Outputs:
            None. Schedules event in priority queue and updates passenger list
            
        Event encoding:
            1 = passenger arrival
            0 = departure from normal queue
            2 = officer screening finished
        """
        arrival_time = now
        distribution = sts.expon(scale=1/self.arrival_rate)
        sampled_interarrival_time = distribution.rvs(size=1)
        arrival_time += sampled_interarrival_time[0] 
        self.curr_arrival_time = arrival_time
        
        self.passenger_counter += 1
        passenger = Passenger(id=self.passenger_counter, arrival_time=arrival_time, p_screen=self.screening_prob)
        self.all_passengers[passenger.id] = passenger 
        
        # Create passenger and schedule arrival
        heapq.heappush(self.priority_q, (arrival_time, (1, None, passenger.id)))
    
    def run_next_service(self):
        """
        Process the next event in the priority queue.
        
        Inputs:
            None (uses internal priority queue)
            
        Outputs:
            None. Updates simulation state and schedules future events
        """
        timestamp, event_info = heapq.heappop(self.priority_q)
        self.now = timestamp

        event_type, queue, passenger_id = event_info

        if event_type == 1:  # arrival
            passenger = self.all_passengers[passenger_id]
            selected_queue = self.find_shortest_queue()
            selected_queue.add_passenger(passenger)

            if selected_queue.being_served_count() == 0:  # if server is free
                passenger = selected_queue.get_next_passenger()
                end_service_time = selected_queue.start_service(passenger, self.now)
                heapq.heappush(self.priority_q, (end_service_time, (0, selected_queue, passenger.id)))

        elif event_type == 0:  # base service finished at a station
            passenger = self.all_passengers[passenger_id]
            
            if passenger.needs_additional_screening:
                # Don't end_service yet, keep server occupied by this passenger
                start_screen = max(passenger.base_service_end_time, self.officer_busy_until)
                duration = sample_truncated_normal(queue.add_mu, queue.add_sigma)
                end_screen = start_screen + duration
                self.officer_busy_until = end_screen

                # final completion of service becomes officer end
                passenger.service_end_time = end_screen

                # schedule officer completion; only then free the station
                heapq.heappush(self.priority_q, (end_screen, (2, queue, passenger.id)))
            else:
                # end service immediately if no additional screening needed
                passenger.service_end_time = passenger.base_service_end_time
                
                # no additional screening: free server and start next
                queue.end_service()
                next_passenger = queue.get_next_passenger()
                if next_passenger:
                    end_service_time = queue.start_service(next_passenger, self.now)
                    heapq.heappush(self.priority_q, (end_service_time, (0, queue, next_passenger.id)))

        elif event_type == 2:  # officer finished screening passenger
            # Now free the station and start the next person, if any
            queue.end_service()
            next_passenger = queue.get_next_passenger()
            if next_passenger:
                end_service_time = queue.start_service(next_passenger, self.now)
                heapq.heappush(self.priority_q, (end_service_time, (0, queue, next_passenger.id)))


                               
    def __repr__(self):
        """String summary of airport queues and screening count."""
        total_waiting = sum(q.waiting_count() for q in self.passenger_queues)
        total_being_served = sum(q.being_served_count() for q in self.passenger_queues)
        
        result = f"Airport at time (minutes) {np.round(self.now, 3)}: {total_waiting} total waiting, {total_being_served} total being served"
        
        for i, q in enumerate(self.passenger_queues):
            result += f"\n  Queue {i+1}: {q.waiting_count()} waiting, {q.being_served_count()} being served."

        additional_screening_count = sum(1 for p in self.all_passengers.values() if p.needs_additional_screening)
        result += f"\n  Additional Screening: {additional_screening_count} needed it so far."
        return result
    
    def print_schedule(self):
        """
        Print current schedule of events and queue states.
        
        Inputs:
            None
            
        Outputs:
            None. Prints to user
        """
        print(repr(self))
        for time, event in sorted(self.priority_q):
            event_type, queue, passenger_id = event
            passenger = self.all_passengers[passenger_id]
            queue_index = self.passenger_queues.index(queue) + 1 if queue else None

            if event_type == 1:  # arrival
                print(f"\t[Arrival]    t={time:.2f}: Passenger {passenger_id} → Queue {queue_index}")
            elif event_type == 0:  # departure from normal queue
                note = " (needs screening)" if passenger.needs_additional_screening else ""
                print(f"\t[Departure]  t={time:.2f}: Passenger {passenger_id} left Queue {queue_index}{note}")
            elif event_type == 2:   # officer finished screening
                print(f"\t[Officer]    t={time:.2f}: Passenger {passenger_id} finished screening")
        print("\n")
