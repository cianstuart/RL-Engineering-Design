# Structural & material Properties

L = 2.0            # Horizontal length to tip (m)
H = 1.0            # Vertical height of upper support (m)
E = 2.1e11         # Young's Modulus (Pa) for steel
A = 1e-4           # Cross-sectional area of rods (m^2)
F_TIP = 10000.0    # Downward load applied at tip (N)

# Parameters used for the parametric truss environment
X_MIN = 0.1
X_MAX = 1.9
Y_MIN = 0.05
Y_MAX = 0.95
STEP_SIZE = 0.025
MAX_STEPS = 150

# RL parameters
TOTAL_TIMESTEPS = 200_000
LOG_DIR = "./data/logs/"
MODEL_DIR = "./data/models/"
EVAL_FREQ = 5_000

PPO_CONFIG_PARAMETRIC = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.05,
    "verbose": 1,
}

PPO_CONFIG_CONSTRUCTIVE = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "verbose": 1,
}