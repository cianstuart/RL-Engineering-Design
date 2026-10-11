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
TOTAL_TIMESTEPS = 500_000
LOG_DIR = "./data/logs/"
MODEL_DIR = "./data/models/"
EVAL_FREQ = 5_000
N_ENVS = 8
SEEDS = [0,1,2]

PPO_CONFIG_PARAMETRIC = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 256,          
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
    "n_steps": 256,          
    "batch_size": 256,        
    "n_epochs": 5,            
    "gamma": 1.0,             
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "target_kl": 0.03,        
    "policy_kwargs": dict(log_std_init=-0.7, net_arch=[128, 128]),  
    "verbose": 1,
}