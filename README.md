# 🚦 T-REX – Traffic Control Environment for Robustness Evaluation under Incidents

T-REX is an open-source simulation framework for training and evaluating **Reinforcement Learning-based Traffic Signal Control (RL-TSC)** under **network-level incident scenarios**. Built on top of [SUMO](https://www.eclipse.org/sumo/), T-REX enables **realistic and reproducible** simulation of dynamic disruptions like accidents, lane blockages, and rerouting behavior to assess the **robustness of control strategies**.

> Designed for researchers and practitioners seeking robust traffic signal control under uncertainty.

📖 **[Documentation (Coming Soon)]** | 🚀 **Quickstart** | 💡 **Use Cases** | 🧪 **Benchmarking Scenarios**

---

## 🎯 Key Features

✅ **Incident-Aware Simulation** – Integrates random or user-defined incidents with duration, location, and lane impact.  
✅ **Realistic Driver Behavior** – Includes rerouting via the Information Comply Model (ICM), speed adaptation, and lane-change reactions.  
✅ **RL-Compatible** – Interfaces with any RL algorithm using [RESCO](https://github.com/TUM-VT/RESCO) and OpenAI Gym.  
✅ **Network-Level Evaluation** – Supports large-scale simulations with dynamic congestion propagation.  
✅ **Modular Design** – Separate Initializer, Deployment, and RL interfaces for flexibility.  
✅ **Robustness Metrics** – Includes learning stability, performance degradation, convergence rate, and AUC comparisons.

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/<your-org>/T-REX.git
cd T-REX

# Set up environment (SUMO + Python)
conda env create -f environment.yml
conda activate trex

# Run a base scenario
python run_simulation.py --scenario grid4x4 --incident False

# Run an incident scenario
python run_simulation.py --scenario grid4x4 --incident True
```

✔ Results stored in `T-REX/results/{scenario_name}/`  
✔ RL agents can be trained using RESCO or your preferred Gym-compatible library

---

## 🧰 Installation

### 1️⃣ Dependencies

- [SUMO](https://www.eclipse.org/sumo/)
- Python ≥ 3.8
- TraCI (Python API for SUMO)
- OpenAI Gym
- RESCO (for RL interaction)

```bash
# Install SUMO (example using apt)
sudo apt install sumo sumo-tools

# Install Python dependencies
pip install -r requirements.txt
```

---

## 🧠 RL Integration

T-REX uses **RESCO** to interface with reinforcement learning agents. The environment exposes:

- **State:** per-intersection traffic observations (e.g., queue length, pressure, incoming flow)
- **Action:** signal phase decisions (discrete)
- **Reward:** delay, throughput, or custom metrics

Supports algorithms like:
- IDQN
- IPPO
- MPLight
- FMA2C
- + any custom method using Gym interface

---

## 🔥 Incident Modeling

T-REX supports two modes of incident deployment:
- **Collision simulation** using real-time vehicle manipulation
- **Virtual blockage** using dummy vehicles (recommended)

Features:
- Configurable duration, location, number of blocked lanes
- Random or deterministic generation
- Anti-teleportation strategies
- Supports multiple simultaneous incidents

---

## 🚘 Driver Behavior Modules

- **Rerouting:** Based on the Information Comply Model (ICM), including awareness via radio, VMS, apps, and observation.
- **Speed adaptation:** Slowdown based on stopping sight distance (SSD)
- **Lane changing:** Enhanced behavior using modified SUMO lane change model (LC2013)

---

## 📊 Robustness Evaluation

T-REX offers multiple metrics beyond average travel time:

| Metric | Description |
|--------|-------------|
| `Performance Degradation (P)` | Relative drop from base to incident scenario |
| `Learning Stability Index (LSI)` | Variation in learning curve |
| `Final Performance Deviation (FPD)` | Drop from best to final episode |
| `Convergence Rate (CR)` | Speed and direction of convergence |
| `Relative AUC Difference (RAUC)` | Difference in total performance over time |

---

## 🧪 Benchmark Networks

T-REX includes five traffic networks:

| Scenario | Description |
|----------|-------------|
| Grid4x4 | 16-intersection synthetic network |
| Cologne Corridor | 3 intersections (TAPAS) |
| Cologne Region | 8 intersections (TAPAS) |
| Ingolstadt Corridor | 7 intersections |
| Ingolstadt Region | 21 intersections (realistic large-scale case)

---

## 📂 Folder Structure

```bash
T-REX/
├── scenarios/            # Network files, traffic demand
├── modules/              # Initializer, deployment, behavior models
├── agents/               # RL-TSC methods and baselines
├── results/              # Output files and logs
├── run_simulation.py     # Main execution script
└── environment.yml       # Conda environment definition
```

---

## 🤝 Contributing

We welcome community contributions!

1. Fork the repo and create a new branch
2. Submit a pull request after testing with:
```bash
python tests/run_all_tests.py
```

---

## 📢 Citation

If you use T-REX in your research, please cite:

```bibtex
@article{your2025trex,
  title={Robustness of Reinforcement Learning-Based Traffic Signal Control under Incidents: A Comparative Study},
  author={Author One, Author Two, Author Three},
  journal={Transportation Research Part C},
  year={2025}
}
```