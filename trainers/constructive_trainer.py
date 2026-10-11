import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import parameters as p
from envs.constructive_truss_env import ConstructiveTrussEnv

def linear_schedule(initial_lr):
    def schedule(progress_remaining):
        return initial_lr * progress_remaining
    return schedule


def train_agent(seed=0):
    # Create directories
    log_path = os.path.join(p.LOG_DIR, "constructive_truss")
    model_path = os.path.join(p.MODEL_DIR, "constructive_truss")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # Initialize parallel environments
    env = make_vec_env(
        ConstructiveTrussEnv,
        n_envs=p.N_ENVS,
        seed=seed,
        env_kwargs={"params": p},
        vec_env_cls=SubprocVecEnv if p.N_ENVS > 1 else DummyVecEnv,
        monitor_dir=log_path,
    )
    eval_env = Monitor(ConstructiveTrussEnv(params=p))

    # Initialize the PPO agent with the LR scheduler
    config = dict(p.PPO_CONFIG_CONSTRUCTIVE)
    config["learning_rate"] = linear_schedule(config["learning_rate"])
    model = PPO(env=env, tensorboard_log=log_path, seed=seed, **config)

    # Set up callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=max(p.EVAL_FREQ // p.N_ENVS, 1),
        n_eval_episodes=1,      
        deterministic=True,
        render=False,
    )

    # Train the agent
    print("Starting training")
    model.learn(total_timesteps=p.TOTAL_TIMESTEPS, callback=[eval_callback])

    # Save the final model
    final_model_path = os.path.join(model_path, "final_model")
    model.save(final_model_path)
    env.close()

    # evaluate the best model
    best = PPO.load(os.path.join(model_path, "best_model.zip"))
    return evaluate_agent(best, ConstructiveTrussEnv(params=p))

def evaluate_agent(model, env):
    print("Evaluating the trained agent")

    # Initialize
    obs, _ = env.reset(seed=83)

    terminated = False
    truncated = False
    step_count = 0
    metrics = {}

    # Run the agent
    while not (terminated or truncated) and step_count < p.MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, metrics = env.step(action)
        step_count += 1

    compliance = metrics.get("compliance", np.nan)
    # Define node positions and bars
    nodes = np.array([[0.0, 0.0], [0.0, p.H], [p.L, 0.0], obs[1:3], obs[3:5]], dtype=float)
    bar_decisions = obs[5:]


    fig, ax = plt.subplots()

    for (nodeA, nodeB), decision in zip(env.possible_connections, bar_decisions):
        x_values = [nodes[nodeA, 0], nodes[nodeB,0]]
        y_values = [nodes[nodeA, 1], nodes[nodeB,1]]
        if decision < 0.5: # dont place
            ax.plot(x_values, y_values, "red", linestyle="--", alpha=0.4)
        else: #place
            ax.plot(x_values, y_values, "green", linewidth=2)

    ax.plot(nodes[:2, 0], nodes[:2, 1], "^", color="gray", markersize=9, label="Fixed supports")
    ax.plot(nodes[2,0], nodes[2,1], ".", color="gray", markersize=12, label="Tip")
    ax.plot(nodes[3:, 0], nodes[3:, 1], ".", color="blue", markersize=12, label="Agent nodes")

    ax.set_xlabel("X-coordinate (m)")
    ax.set_ylabel("Y-coordinate (m)")
    ax.set_title(f"Constructive truss design. Compliance = {compliance:.2f} J")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    import sys

    if "--eval" in sys.argv:
        # Load the trained model and evaluate
        model_path = os.path.join(p.MODEL_DIR, "constructive_truss", "best_model.zip")
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Please train the agent first.")
            sys.exit(1)

        raw_env = ConstructiveTrussEnv(params=p)
        model = PPO.load(model_path, env=raw_env)

        evaluate_agent(model, raw_env)
    else:
        train_agent()