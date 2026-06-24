1. Can a single TSC method optimize the delay/queue for all types of flows? Intersection level, network level

2. Can a single TSC method obtain the optimal signal cycle length
on the basis of the relative proportion of people using different modes? Intersection level, network level

-  Benchmark against baselines when evaluated on multimodal metrics
    - Vs STOCHASTIC/MAXWAVE/MAXPRESSURE and fixed time: per-mode average wait, 90th-percentile wait (captures fairness tail), throughput (vehicles/bikes/peds served per hour), and queue stability (variance over time)
    - vs other RL methods, idem

- Can the TSC method devised and trained for a single scenario be easily adapted for a different one? (e.g. Does a multimodal IDQN policy trained on one scenario transfer to others with different lane configurations, phase counts, or modal share?)
    - Train on kbh_j1_442m, evaluate zero-shot on 2–3 structurally different intersections/network and compare against training each intersection from scratch. Metric: performance retention ratio (transferred reward / from-scratch reward) per mode.

- Does the lane-wise state architecture (vs. the original Conv2d) generalize the benefits, or was it specific to this network's feature heterogeneity (multimodality)?
    - Train both on the car only and the multimodal, 

