# CPU Scheduling Simulator

A simple Python-based simulator for CPU scheduling algorithms. This project implements:

- **SRTF** – Shortest Remaining Time First (preemptive SJF)  
- **MLFQ** – A three-level Multi-Level Feedback Queue (RR with quanta of 8 and 16, then FCFS)

It generates random (or edge-case) processes, runs both schedulers, and reports key metrics:

- Average Waiting Time (AWT)  
- Average Turnaround Time (ATT)  
- CPU Utilization (%)  
- Throughput (processes per time unit)  

---


## Features

- **Process generators**  
  - Small batch (5 processes, fixed seed)  
  - Large batch (30 processes, random arrivals/bursts)  
  - Identical-arrival/burst edge case  
  - Mixed long/short burst edge case  

- **CSV output**  
  - `processes_<mode>.csv` — list of generated processes  
  - `metrics_<mode>.csv`   — performance metrics for SRTF and MLFQ  

- **Console menu**  
  Choose between modes (`s`, `l`, `ei`, `er`) and view results in real time.

---

## Prerequisites

- Python 3.7+  
- No external dependencies (only Python’s standard library)

---

## Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/cpu-scheduler-sim.git
   cd cpu-scheduler-sim

2.	Run the simulator
python scheduler.py

3. Follow on-screen prompts:

   - **Enter one of:**
     - `s` – small batch  
     - `l` – large batch  
     - `ei` – edge case: identical arrival & burst  
     - `er` – edge case: mixed long & short bursts  
     - `q` – quit  

   - After running, you’ll see per-algorithm metrics printed and CSV files written in the project folder.
---

## File Overview

- `scheduler.py`           – Main simulation script  
- `metrics_<mode>.csv`     – Output metrics  
- `processes_<mode>.csv`   – Generated process lists  
- `README.md`              – This file


  
## License

This project is released under the MIT License. Feel free to use and adapt as needed.
