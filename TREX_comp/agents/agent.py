import hashlib
import os
import torch


class Agent(object):
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda:0"
            print("Using GPU: {}".format(torch.cuda.get_device_name(0)))
        else:
            device = "cpu"
            print("Using CPU")
        self.device = torch.device(device)

    def act(self, observation):
        raise NotImplementedError

    def observe(self, observation, reward, done, info):
        raise NotImplementedError


def _safe_model_path(log_dir, agent_id, prefix="agent_", max_len=200):
    # Sanitize and shorten filenames to avoid OS path-length limits.
    safe_id = str(agent_id).replace(os.sep, "_").replace("..", "_")
    if os.altsep:
        safe_id = safe_id.replace(os.altsep, "_")
    base_name = f"{prefix}{safe_id}"
    if len(base_name) > max_len:
        digest = hashlib.sha1(base_name.encode("utf-8")).hexdigest()[:10]
        base_name = f"{prefix}{safe_id[: max_len - len(prefix) - 11]}_{digest}"
    return os.path.join(log_dir, base_name)

class IndependentAgent(Agent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__()
        self.config = config
        self.agents = dict()

    def act(self, observation):
        acts = dict()
        for agent_id in observation.keys():
            acts[agent_id] = self.agents[agent_id].act(observation[agent_id])
        return acts

    def observe(self, observation, reward, done, info):
        for agent_id in observation.keys():
            self.agents[agent_id].observe(observation[agent_id], reward[agent_id], done, info)
            if done:
                if info['eps'] % self.config['save_freq'] == 0:
                    save_path = _safe_model_path(self.config['log_dir'], agent_id)
                    self.agents[agent_id].save(save_path)
                # if info['eps'] == 34:
                #     self.agents[agent_id].save(self.config['log_dir']+'agent_'+agent_id)


class SharedAgent(Agent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__()
        self.config = config
        self.agent = None
        self.valid_acts = None
        self.reverse_valid = None

    def act(self, observation):
        if self.reverse_valid is None and self.valid_acts is not None:
            self.reverse_valid = dict()
            for signal_id in self.valid_acts:
                self.reverse_valid[signal_id] = {v: k for k, v in self.valid_acts[signal_id].items()}

        batch_obs = [observation[agent_id] for agent_id in observation.keys()]
        if self.valid_acts is None:
            batch_valid = None
            batch_reverse = None
        else:
            batch_valid = [self.valid_acts.get(agent_id) for agent_id in
                           observation.keys()]
            batch_reverse = [self.reverse_valid.get(agent_id) for agent_id in
                          observation.keys()]

        batch_acts = self.agent.act(batch_obs,
                                valid_acts=batch_valid,
                                reverse_valid=batch_reverse)
        acts = dict()
        for i, agent_id in enumerate(observation.keys()):
            acts[agent_id] = batch_acts[i]
        return acts

    def observe(self, observation, reward, done, info):
        batch_obs = [observation[agent_id] for agent_id in observation.keys()]
        batch_rew = [reward[agent_id] for agent_id in observation.keys()]
        batch_done = [done]*len(batch_obs)
        batch_reset = [False]*len(batch_obs)
        self.agent.observe(batch_obs, batch_rew, batch_done, batch_reset)
        if done:
            if info['eps'] % self.config['save_freq'] == 0:
                save_path = _safe_model_path(self.config['log_dir'], "shared", prefix="agent_")
                self.agent.save(save_path)
