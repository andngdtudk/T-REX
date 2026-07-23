-  Benchmark against baselines when evaluated on multimodal metrics, and on sum of modes
    - Vs STOCHASTIC/MAXWAVE/MAXPRESSURE and fixed time: per-mode average wait, 90th-percentile wait (captures fairness tail), throughput (vehicles/bikes/peds served per hour), and queue stability (variance over time)
    - vs other RL methods, idem

- Benchmark against baselines when using the relative proportion of people using different modes -> weights

- Can the TSC method devised and trained for a single scenario be easily adapted for a different one? (different lane configurations, phase counts, or modal share?)
    - Train on kbh_j1_442m, evaluate zero-shot on structurally different intersections/network and compare against training each intersection from scratch. Measure performance retention ratio (transferred reward / from-scratch reward) per mode.

- IDQN: Does the lane-wise state architecture (vs. the original Conv2d) generalize the benefits, or was it specific to this network's feature heterogeneity (multimodality)?
    - Train both on the car only and the multimodal, compare performance

- Does giving pedestrians phase-level visibility change vehicle behaviors significantly?
    - MPLight: We are adding to the FRAP ped pressure and bike queues, that makes busy pedestrian/cyclist phases atractive to the agent regardless of car queues: compare with weight for other modes = 0
    - IDQN: idem, especially when weights have so much influence
    - Both: Sweeping weights and finding Pareto frontier