import gymnasium as gym
from gymnasium import spaces
import numpy as np
import parameters as default_params
from fem.truss_2d_constructive import Truss2DConstructive

class ConstructiveTrussEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, params=default_params):
        super().__init__()
        self.params = params
        self.truss = Truss2DConstructive(params = self.params)

        # Define possible connections for the truss elements
        self.possible_connections = np.array([
            (0,2),                  # Bottom-left to Tip 
            (0, 3), (1, 3),         # Fixed supports to Node 3
            (0, 4), (1, 4),         # Fixed supports to Node 4
            (3, 4),                 # Node 3 to Node 4
            (2, 3), (2, 4)          # Node 3 & 4 to Tip (Node 2)
        ], dtype=int)
        self.num_connections = len(self.possible_connections)

        # Steps per episode
        self.build_steps = 2 + self.num_connections  # 2 steps for node placement + steps for each connection

        # Continuous action space (these can be used: they will be mapped to coordinate for step 1 and step 2, and for bar placement only the first action will be used)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32)

        
        # Observation space : keeps track of current step (optimal policy will differ for each step), placed positions of nodes, and  bar decisions so far (1 = placed, 0 = not placed)
        obs_low = np.array([0.0, params.X_MIN, params.Y_MIN, params.X_MIN, params.Y_MIN] + [0.0]*self.num_connections, dtype=np.float32)
        obs_high = np.array([float(self.build_steps), params.X_MAX, params.Y_MAX, params.X_MAX, params.Y_MAX] + [1.0]*self.num_connections, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.current_step = 0
        self.pos1 = np.array([0.5*(self.params.X_MIN + self.params.X_MAX), 0.5*(self.params.Y_MIN + self.params.Y_MAX)], dtype = np.float32)
        self.pos2 = np.array([0.5*(self.params.X_MIN + self.params.X_MAX), 0.5*(self.params.Y_MIN + self.params.Y_MAX)], dtype = np.float32)
        self.bar_decisions = np.zeros(self.num_connections, dtype=np.float32)

    def scale_action(self, action):
        """
        Scales the action from [-1, 1] to the coordinate space.
        """
        x_scaled = np.clip(((action[0] + 1) / 2) * (self.params.X_MAX - self.params.X_MIN) + self.params.X_MIN, self.params.X_MIN, self.params.X_MAX)
        y_scaled = np.clip(((action[1] + 1) / 2) * (self.params.Y_MAX - self.params.Y_MIN) + self.params.Y_MIN, self.params.Y_MIN, self.params.Y_MAX)
        return np.array([x_scaled, y_scaled], dtype=np.float32)
    
    def get_observation(self):
        return np.concatenate([np.array([float(self.current_step)], dtype=np.float32),self.pos1,self.pos2,self.bar_decisions], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.pos1 = np.array([0.5*(self.params.X_MIN + self.params.X_MAX), 0.5*(self.params.Y_MIN + self.params.Y_MAX)], dtype = np.float32)
        self.pos2 = np.array([0.5*(self.params.X_MIN + self.params.X_MAX), 0.5*(self.params.Y_MIN + self.params.Y_MAX)], dtype = np.float32)
        self.bar_decisions = np.zeros(self.num_connections, dtype=np.float32)

        metrics = {}
        return self.get_observation(), metrics

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward = 0.0
        terminated = False
        truncated = False
        metrics = {}

        if self.current_step == 0:
            self.pos1 = self.scale_action(action)
        elif self.current_step == 1:
            self.pos2 = self.scale_action(action)
        else:
            bar_index = self.current_step - 2
            if action[0] > 0:
                self.bar_decisions[bar_index] = 1.0
                reward -= 0.1
            else:
                self.bar_decisions[bar_index] = 0.0

        self.current_step += 1

        if self.current_step >= self.build_steps:
            terminated = True
            active_connections = self.possible_connections[self.bar_decisions > 0.5]
            try:
                fem_result = self.truss.solve(self.pos1, self.pos2, active_connections)
                compliance = fem_result.compliance
            except Exception:
                compliance = np.nan

            if  not np.isfinite(compliance) or compliance <= 0:
                reward += -15.0
            else:
                compliance_ref = 40.0
                reward += -np.log(compliance / compliance_ref)                   
                metrics["compliance"] = float(compliance)

        return self.get_observation(), reward, terminated, truncated, metrics

  