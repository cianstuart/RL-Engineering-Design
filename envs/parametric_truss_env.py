import gymnasium as gym
from gymnasium import spaces
import numpy as np
import parameters as default_params
from fem.truss_2d_parametric import Truss2DParametric

class ParametricTrussEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    def __init__(self, params=default_params):
        super().__init__()
        self.params = params
        self.truss = Truss2DParametric(params = self.params)

        # Discrete action space
        self.action_space = spaces.Discrete(5) # 5 discrete actions: Up, Down, Left, Right, Stay
        step_size = self.params.STEP_SIZE
        self.action_mapping = {
            0: (0.0, step_size),    # Up
            1: (0.0, -step_size),   # Down
            2: (-step_size, 0.0),   # Left
            3: (step_size, 0.0),     # Right
            4: (0.0, 0.0)           # Stay
        }

        # Observation space
        x_min = self.params.X_MIN
        x_max = self.params.X_MAX
        y_min = self.params.Y_MIN
        y_max = self.params.Y_MAX
        self.observation_space = spaces.Box(low=np.array([x_min, y_min, 0.0], dtype=np.float32), 
                                            high=np.array([x_max, y_max, 1000.0], dtype=np.float32), 
                                            dtype=np.float32)

        # Train params
        self.max_steps = self.params.MAX_STEPS
        self.current_step = 0
        self.current_pos = None
        self.prev_compliance = None

    def get_observation(self):
        return np.array([self.current_pos[0], self.current_pos[1], self.prev_compliance], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Random initial position as a multiple of the step size
        x_raw = self.np_random.uniform(self.params.X_MIN, self.params.X_MAX)
        y_raw = self.np_random.uniform(self.params.Y_MIN, self.params.Y_MAX)
        x_init = np.round(x_raw/self.params.STEP_SIZE) * self.params.STEP_SIZE
        y_init = np.round(y_raw/self.params.STEP_SIZE) * self.params.STEP_SIZE

        self.current_pos = np.array([x_init, y_init], dtype=np.float32)
       
        # Calculate initial compliance
        fem_result = self.truss.solve(self.current_pos[0], self.current_pos[1])
        compliance = fem_result.compliance

        # Resample if neccessary 
        while np.isnan(compliance) or np.isinf(compliance):
            x_raw = self.np_random.uniform(self.params.X_MIN, self.params.X_MAX)
            y_raw = self.np_random.uniform(self.params.Y_MIN, self.params.Y_MAX)
            x_init = np.round(x_raw/self.params.STEP_SIZE) * self.params.STEP_SIZE
            y_init = np.round(y_raw/self.params.STEP_SIZE) * self.params.STEP_SIZE
            self.current_pos = np.array([x_init, y_init], dtype=np.float32)
            fem_result = self.truss.solve(self.current_pos[0], self.current_pos[1])
            compliance = fem_result.compliance

        self.prev_compliance = compliance
        metrics = {"Compliance": compliance, 
                   "Max stress": float(np.max(np.abs(fem_result.stresses))), 
                   "Tip displacement": float(fem_result.displacements[5])
                   }
        return self.get_observation(), metrics

    def step(self, action):
        action = int(action)
        # Apply step and clip
        dx,dy = self.action_mapping[action]
        new_x = np.clip(self.current_pos[0] + dx, self.observation_space.low[0], self.observation_space.high[0])
        new_y = np.clip(self.current_pos[1] + dy, self.observation_space.low[1], self.observation_space.high[1])
        self.current_pos = np.array([new_x, new_y], dtype=np.float32)

        # Solve FEM
        fem_result = self.truss.solve(new_x, new_y)

        if np.isnan(fem_result.compliance) or np.isinf(fem_result.compliance):
            reward = -100.0  
            terminated = True
            metrics = {"Compliance": np.nan, 
                        "Max stress": np.nan, 
                        "Tip displacement": np.nan
                        }
        else:
            reward = float(np.log(self.prev_compliance) - np.log(fem_result.compliance)) # Change in log compliance
            self.prev_compliance = fem_result.compliance
            terminated = False
            metrics = {"Compliance": fem_result.compliance, 
                        "Max stress": float(np.max(np.abs(fem_result.stresses))), 
                        "Tip displacement": float(fem_result.displacements[5])
                        }

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        else:
            truncated = False

        return self.get_observation(), reward, terminated, truncated, metrics