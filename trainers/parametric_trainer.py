import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import parameters as p
from envs.parametric_truss_env import ParametricTrussEnv

def train_agent():
    # Create directories
    log_path = os.path.join(p.LOG_DIR, "parametric_truss")
    model_path = os.path.join(p.MODEL_DIR, "parametric_truss")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # Initialize the environment
    raw_env = ParametricTrussEnv(params=p)
    env = Monitor(raw_env, log_path)

    # Initialize the PPO agent
    model = PPO(
        env=env,
        tensorboard_log=log_path,
        policy=p.PPO_CONFIG_PARAMETRIC["policy"],
        learning_rate=p.PPO_CONFIG_PARAMETRIC["learning_rate"],
        n_steps=p.PPO_CONFIG_PARAMETRIC["n_steps"],
        batch_size=p.PPO_CONFIG_PARAMETRIC["batch_size"],
        n_epochs=p.PPO_CONFIG_PARAMETRIC["n_epochs"],
        gamma=p.PPO_CONFIG_PARAMETRIC["gamma"],
        gae_lambda=p.PPO_CONFIG_PARAMETRIC["gae_lambda"],
        clip_range=p.PPO_CONFIG_PARAMETRIC["clip_range"],
        ent_coef=p.PPO_CONFIG_PARAMETRIC["ent_coef"],
        verbose=p.PPO_CONFIG_PARAMETRIC["verbose"],
    )

    # Set up callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=p.EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    # Train the agent
    print("Starting training")
    model.learn(total_timesteps=p.TOTAL_TIMESTEPS, callback=[eval_callback])

    # Save the final model
    final_model_path = os.path.join(model_path, "final_model")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to: {final_model_path}")

    # Evaluate the trained agent
    evaluate_agent(model, raw_env)

def evaluate_agent(model, env):
    print("Evaluating the trained agent")

    # Initialize
    obs, metrics = env.reset(seed=83)
    start_pos = obs.copy()
    start_compliance = metrics["Compliance"]
    print(f"Start Position: {start_pos}, Start Compliance: {start_compliance}")

    trajectory = [start_pos]
    compliances = [start_compliance]

    terminated = False
    truncated = False
    step_count = 0

    # Run the agent
    while not (terminated or truncated) and step_count < p.MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, metrics = env.step(action)

        trajectory.append(obs.copy())
        compliances.append(metrics["Compliance"])
        step_count += 1

    # Evaluate
    final_pos = obs.copy()
    final_compliance = metrics["Compliance"]
    print(f"Final Position: {final_pos}, Final Compliance: {final_compliance}")

    compliance_reduction = (start_compliance - final_compliance)/start_compliance * 100
    print(f"Compliance Reduction: {compliance_reduction:.2f}%")

    # Plot trajectory
    plot_trajectory(trajectory, compliances)

def plot_trajectory(trajectory, compliances):
    coordinates = np.array(trajectory)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plot 1: middle node trajectory
    ax1.plot(coordinates[:, 0], coordinates[:, 1], "ro-", linewidth=1.5, markersize=4, label="Path")
    ax1.plot(coordinates[0, 0], coordinates[0, 1], "go", markersize=9, label="Start")
    ax1.plot(coordinates[-1, 0], coordinates[-1, 1], "bs", markersize=9, label="Final")

    ax1.set_xlim(p.X_MIN - 0.05, p.X_MAX + 0.05)
    ax1.set_ylim(p.Y_MIN - 0.05, p.Y_MAX + 0.05)
    ax1.set_xlabel("X-coordinate (m)")
    ax1.set_ylabel("Y-coordinate (m)")
    ax1.set_title("Middle node trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # plot 2: compliance curve
    ax2.plot(compliances, "b-x", linewidth=1.5, markersize=5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Compliance (J)")
    ax2.set_title("Compliance curve")
    ax2.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    import sys

    if "--eval" in sys.argv:
        # Load the trained model and evaluate
        model_path = os.path.join(p.MODEL_DIR, "parametric_truss", "best_model.zip")
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Please train the agent first.")
            sys.exit(1)

        raw_env = ParametricTrussEnv(params=p)
        model = PPO.load(model_path, env=raw_env)

        evaluate_agent(model, raw_env)
    else:
        train_agent()