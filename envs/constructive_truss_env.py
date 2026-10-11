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

    def orientation(self,a,b,c):
        """
        Determines wheter point c lies to the left or right of a line segment going from a to b
        """
        ab_x = b[0] - a[0]
        ab_y = b[1] - a[1]
        ac_x = c[0] - a[0]
        ac_y = c[1] - a[1]
        cross_2d = ab_x*ac_y - ab_y*ac_x

        if cross_2d > 0: # c is on the left
            return 1
        if cross_2d < 0: # c is on the right
            return -1
        else: # c is on the line
            return 0

    def crossing_segments(self,p1,p2,p3,p4):
        """
        Returns true if the segment p1-p2 crosses p3-p4. 
        They cross if p3 and p4 are on opposite sides of segment p1-p2 AND
        p1 and p2 are on opposite sides of segment p3-p4
        """
        # w.r.t. p1-p2
        side_p3 = self.orientation(p1,p2,p3)
        side_p4 = self.orientation(p1,p2,p4)
        if side_p3*side_p4 < 0:
            p3p4_opposite = True
        else: 
            p3p4_opposite = False

        # w.r.t. to p3-p4
        side_p1 = self.orientation(p3,p4,p1)
        side_p2 = self.orientation(p3,p4,p2)
        if side_p1*side_p2 < 0:
            p1p2_opposite = True
        else: 
            p1p2_opposite = False

        return p1p2_opposite and p3p4_opposite

    def valid_truss(self,nodes,connections):
        """
        Determines whether a truss is valid i.e. has no crossing bars. Returns true if valid.
        """
        n_bars = len(connections)
        i = 0
        while i < n_bars:
            node_a, node_b = connections[i] # first pair
            j = i + 1
            while j < n_bars:
                node_c, node_d = connections[j] # second pair
                # check if they share a node (crossing is impossible)
                share_node = len({node_a, node_b} & {node_c, node_d}) > 0
                if not share_node:
                    if self.crossing_segments(nodes[node_a], nodes[node_b], nodes[node_c], nodes[node_d]):
                        return False
                j += 1
            i += 1
        return True
    
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
        min_seperation = 0.15 #seperation between the agent nodes
        comvol_ref = 0.03 

        if self.current_step == 0:
            self.pos1 = self.scale_action(action)
        elif self.current_step == 1:
            self.pos2 = self.scale_action(action)
        else:
            bar_index = self.current_step - 2
            if action[0] > 0:
                self.bar_decisions[bar_index] = 1.0
            else:
                self.bar_decisions[bar_index] = 0.0

        self.current_step += 1

        if self.current_step >= self.build_steps:
            terminated = True
            active_connections = self.possible_connections[self.bar_decisions > 0.5]
            nodes = self.truss.get_nodes(self.pos1, self.pos2)

            if np.linalg.norm(self.pos1 - self.pos2) < min_seperation:
                reward += -5                   # agent nodes too close
            elif not self.valid_truss(nodes, active_connections):
                reward += -5                     # crossing bars
            else:
                try:
                    fem_result = self.truss.solve(self.pos1, self.pos2, active_connections)
                    compliance = fem_result.compliance
                except Exception:
                    compliance = np.nan

                if not np.isfinite(compliance) or compliance <= 0:
                    reward += -5                 
                else:
                    conA, conB = active_connections[:, 0], active_connections[:, 1]
                    volume = self.params.A * np.sum(np.linalg.norm(nodes[conA] - nodes[conB], axis=1))
                    reward += -np.log(compliance * volume / comvol_ref)
                    metrics["compliance"] = float(compliance)

        return self.get_observation(), reward, terminated, truncated, metrics

  