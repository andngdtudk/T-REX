from TREX_comp.agents.agent import IndependentAgent, Agent


DEFAULT_HOLD_SECONDS = 20


class FIXEDTIME(IndependentAgent):
    """Fixed-time baseline controller.

    Cycles each signal round-robin through its own green phases (0, 1, 2, ...,
    back to 0), holding every phase for a fixed amount of simulated time
    (default 20 seconds) regardless of demand. It never looks at the state
    observation and never learns from the reward -- both are accepted (the
    env always computes and passes them) purely to satisfy the same
    act()/observe() interface every other agent uses; this agent's own
    logic touches neither.

    Useful as a naive baseline to compare learned/rule-based agents against.
    """
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)

        # step_length = seconds of simulated time advanced per act()/env.step()
        # decision. Threaded in from map_config via run_trial() in main.py.
        step_length = config.get('step_length', 1)
        hold_seconds = config.get('fixed_phase_seconds', DEFAULT_HOLD_SECONDS)

        for key in obs_act:
            num_phases = obs_act[key][1]
            self.agents[key] = FixedTimeAgent(num_phases, step_length, hold_seconds=hold_seconds)


class FixedTimeAgent(Agent):
    def __init__(self, num_phases, step_length, hold_seconds=DEFAULT_HOLD_SECONDS):
        super().__init__()
        self.num_phases = max(1, int(num_phases))

        # Number of consecutive decisions to hold a phase for, so that it is
        # actually held for `hold_seconds` of simulated time no matter what
        # --step_length the map uses (e.g. step_length=10 -> hold_steps=2;
        # step_length=5 -> hold_steps=4; both hold for 20s of sim time).
        if step_length and step_length > 0:
            self.hold_steps = max(1, round(hold_seconds / step_length))
        else:
            self.hold_steps = 1

        self.current_phase = 0
        self._decisions_on_phase = 0

    def act(self, observation):
        # `observation` is intentionally ignored -- fixed-time control does
        # not use state information of any kind.
        phase = self.current_phase

        self._decisions_on_phase += 1
        if self._decisions_on_phase >= self.hold_steps:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self._decisions_on_phase = 0

        return phase

    def observe(self, observation, reward, done, info):
        # No learning of any kind -- reward is intentionally ignored.
        # Reset phase-timing state at episode boundaries so every episode
        # starts from phase 0 on a clean 20s (or configured) hold, rather
        # than resuming mid-hold from wherever the last episode left off.
        if done:
            self.current_phase = 0
            self._decisions_on_phase = 0

    def save(self, path):
        # Nothing to persist -- there is no learned state.
        pass