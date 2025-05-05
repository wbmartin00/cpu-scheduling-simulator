import random
import csv
import copy
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

@dataclass
class PCB:
    pid: int
    arrival_time: int
    burst_time: int
    remaining_time: int = field(init=False)
    start_time: Optional[int] = None
    completion_time: Optional[int] = None

    def __post_init__(self):
        # Initialize remaining time to burst time
        self.remaining_time = self.burst_time


def generate_processes(n: int, max_arrival: int, max_burst: int, seed: Optional[int] = None) -> List[PCB]:
    """
    Generate a list of n processes with random arrival and burst times.
    If `seed` is provided, the sequence is reproducible; otherwise, each call produces different random processes.

    :param n: Number of processes
    :param max_arrival: Maximum arrival time (inclusive)
    :param max_burst: Maximum burst time (inclusive)
    :param seed: Optional random seed for reproducibility (useful for deterministic tests)
    """
    if seed is not None:
        random.seed(seed)
    procs = []
    for i in range(1, n + 1):
        arrival = random.randint(0, max_arrival)
        burst = random.randint(1, max_burst)
        procs.append(PCB(pid=i, arrival_time=arrival, burst_time=burst))
    return procs


def simulate_srtf(processes: List[PCB]) -> dict:
    """
    Shortest Remaining Time First (preemptive SJF) scheduling simulation.
    Returns performance metrics as a dict.
    """
    # Deep copy so original list isn't modified
    procs = sorted(copy.deepcopy(processes), key=lambda p: p.arrival_time)
    time = 0
    completed = 0
    n = len(procs)
    cpu_busy = 0

    # Run until all processes are completed
    while completed < n:
        # Find all arrived and unfinished processes
        available = [p for p in procs if p.arrival_time <= time and p.remaining_time > 0]
        if available:
            # Pick process with smallest remaining time
            curr = min(available, key=lambda p: p.remaining_time)
            if curr.start_time is None:
                curr.start_time = time
            # Execute one time unit
            curr.remaining_time -= 1
            cpu_busy += 1
            time += 1
            # Check for completion
            if curr.remaining_time == 0:
                curr.completion_time = time
                completed += 1
        else:
            # No process is ready; advance time
            time += 1

    # Compute metrics
    total_wait = sum((p.completion_time - p.arrival_time - p.burst_time) for p in procs)
    total_turn = sum((p.completion_time - p.arrival_time) for p in procs)
    makespan = time
    awt = total_wait / n
    att = total_turn / n
    cpu_util = (cpu_busy / makespan) * 100
    throughput = n / makespan

    return {
        'Algorithm': 'SRTF',
        'AWT': awt,
        'ATT': att,
        'CPU_Utilization': cpu_util,
        'Throughput': throughput
    }


def simulate_mlfq(processes: List[PCB], quanta: List[int] = [8, 16]) -> dict:
    """
    Multi-Level Feedback Queue scheduling simulation with 3 levels:
      - Q0: Round-robin, quantum = quanta[0]
      - Q1: Round-robin, quantum = quanta[1]
      - Q2: FCFS until completion
    Returns performance metrics as a dict.
    """
    procs = sorted(copy.deepcopy(processes), key=lambda p: p.arrival_time)
    time = 0
    cpu_busy = 0
    completed = 0
    n = len(procs)

    # Queues for each level
    q0, q1, q2 = deque(), deque(), deque()
    curr: Optional[PCB] = None
    curr_q = None
    quantum_counter = 0

    while completed < n:
        # Add new arrivals to Q0
        for p in procs:
            if p.arrival_time == time:
                q0.append(p)

        # Preempt if a higher-level job arrives
        if curr is not None and curr_q > 0 and q0:
            # Put current back to front of its queue
            if curr_q == 1:
                q1.appendleft(curr)
            else:
                q2.appendleft(curr)
            curr = None
            quantum_counter = 0

        # Schedule if CPU idle
        if curr is None:
            if q0:
                curr = q0.popleft()
                curr_q = 0
                quantum_counter = 0
            elif q1:
                curr = q1.popleft()
                curr_q = 1
                quantum_counter = 0
            elif q2:
                curr = q2.popleft()
                curr_q = 2
                quantum_counter = 0
            else:
                # Idle
                time += 1
                continue

            if curr.start_time is None:
                curr.start_time = time

        # Execute one time unit
        curr.remaining_time -= 1
        cpu_busy += 1
        quantum_counter += 1
        time += 1

        # Check for completion
        if curr.remaining_time == 0:
            curr.completion_time = time
            completed += 1
            curr = None
            quantum_counter = 0
        else:
            # Quantum expiration handling
            if curr_q == 0 and quantum_counter == quanta[0]:
                q1.append(curr)
                curr = None
                quantum_counter = 0
            elif curr_q == 1 and quantum_counter == quanta[1]:
                q2.append(curr)
                curr = None
                quantum_counter = 0
            # Q2 runs until completion

    # Compute metrics
    total_wait = sum((p.completion_time - p.arrival_time - p.burst_time) for p in procs)
    total_turn = sum((p.completion_time - p.arrival_time) for p in procs)
    makespan = time
    awt = total_wait / n
    att = total_turn / n
    cpu_util = (cpu_busy / makespan) * 100
    throughput = n / makespan

    return {
        'Algorithm': 'MLFQ',
        'AWT': awt,
        'ATT': att,
        'CPU_Utilization': cpu_util,
        'Throughput': throughput
    }


def write_metrics_to_csv(metrics_list: List[dict], filename: str = 'metrics.csv'):
    """
    Write a list of metric dicts to a CSV file for easy import into R.
    """
    # Use all keys present in any metric dict for the header
    fieldnames = set()
    for m in metrics_list:
        fieldnames.update(m.keys())
    fieldnames = list(fieldnames)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m)


def write_processes_to_csv(processes: List[PCB], filename: str = "processes.csv"):
    """
    Write process list to CSV for inspection.
    """
    fieldnames = ['pid', 'arrival_time', 'burst_time']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in processes:
            writer.writerow({'pid': p.pid, 'arrival_time': p.arrival_time, 'burst_time': p.burst_time})


# --- Edge-case process generators ---
def generate_identical_processes(n=5, arrival=0, burst=5):
    """All processes arrive at same time and have same burst."""
    return [PCB(pid=i+1, arrival_time=arrival, burst_time=burst) for i in range(n)]


def generate_mixed_processes(n=10, max_arrival=10, max_burst=20, seed=None):
    """
    Generates a mix: half with same arrival/burst, half random.
    """
    if seed is not None:
        random.seed(seed)
    fixed = [PCB(pid=i+1, arrival_time=0, burst_time=max_burst) for i in range(n//2)]
    randoms = [
        PCB(pid=n//2+i+1,
            arrival_time=random.randint(0, max_arrival),
            burst_time=random.randint(1, max_burst))
        for i in range(n - n//2)
    ]
    return fixed + randoms


def main():
    print("CPU Scheduling Simulator")
    max_arrival = 10
    max_burst = 20
    print("Options:")
    print("  s  - Small batch (5 processes)")
    print("  l  - Large batch (30 processes)")
    print("  ei - Edge case: identical arrival & burst times")
    print("  er - Edge case: mixed long & short bursts")
    print("  q  - Quit")
    while True:
        choice = input("\nEnter choice (s/l/ei/er/q): ").strip().lower()
        if choice == 'q':
            print("Exiting.")
            break
        elif choice == 's':
            procs = generate_processes(n=5, max_arrival=max_arrival, max_burst=max_burst, seed=1)
            label = "Small batch"
        elif choice == 'l':
            procs = generate_processes(n=30, max_arrival=max_arrival, max_burst=max_burst, seed=None)
            label = "Large batch"
        elif choice == 'ei':
            procs = generate_identical_processes(n=5, arrival=0, burst=max_burst // 2)
            label = "Edge identical"
        elif choice == 'er':
            procs = generate_mixed_processes(n=10, max_arrival=max_arrival, max_burst=max_burst, seed=None)
            label = "Edge mixed"
        else:
            print("Invalid option.")
            continue

        # Run schedulers
        metrics = [
            simulate_srtf(procs),
            simulate_mlfq(procs)
        ]
        # Filenames
        metrics_file = f"metrics_{choice}.csv"
        procs_file = f"processes_{choice}.csv"
        # Write CSVs
        write_metrics_to_csv(metrics, metrics_file)
        write_processes_to_csv(procs, procs_file)
        # Display results
        print(f"\n{label} test results:")
        for m in metrics:
            print(m)
        print(f"Metrics saved to {metrics_file}")
        print(f"Processes saved to {procs_file}\n")

if __name__ == "__main__":
    main()