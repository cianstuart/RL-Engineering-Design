import argparse
import sys
import numpy as np

from gymnasium.utils.env_checker import check_env
import parameters as default_params

from envs.parametric_truss_env import ParametricTrussEnv
from envs.constructive_truss_env import ConstructiveTrussEnv

def run_api_check(env,name):
    """Run the Gymnasium API compliance checker."""
    try:
        check_env(env.unwrapped, skip_render_check=True)
        print(f"\n[{name}] PASSED all Gymnasium API checks!\n")
    except Exception as error:
        print(f"\n[{name}] FAILED API checks:\n{error}\n")
        sys.exit(1)

def validate_parametric():
    # Initialize the environment
    env = ParametricTrussEnv(params=default_params)
    run_api_check(env, "parametric")


    # Perform a few steps to check outputs
    action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Stay"}
    obs, metrics = env.reset(seed=76)
    print(f"Reset position (x, y): {obs}")
    print(f"Initial metrics: {metrics}\n")

    for step_idx in range(1, 6):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, metrics = env.step(action)

        print(f"Step {step_idx} | Action: {action} ({action_names[action]})")
        print(f"  Obs: {obs} | Reward: {reward:+.6f}")
        print(f"  Terminated: {terminated} | Truncated: {truncated}")
        print(f"  Metrics: {metrics}\n")

        if terminated or truncated:
            obs, metrics = env.reset()

    env.close()

def validate_constructive():
    env = ConstructiveTrussEnv(params=default_params)
    run_api_check(env, "constructive")
 
    # check if the the reset obs lies within the observation space
    obs, _ = env.reset(seed=76)
    print(f"Initial obs: {obs}")
    if not env.observation_space.contains(obs):
        print("FAILED: initial observation is outside the observation space\n")
        sys.exit(1)
 
    # do random episodes and check rewards, lengths
    returns, lengths = [], []
    n_ep = 200
    for ep in range(n_ep):
        obs, _ = env.reset(seed=ep)
        done = False
        total = 0.0 
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if not env.observation_space.contains(obs):
                print(f"FAILED: observation out of bounds at episode {ep}, step {steps}: {obs}\n")
                sys.exit(1)
            total += reward
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)

    # check whether any episodes stopped early
    expected_length = env.build_steps
    print(f"\nPeformed ({n_ep} episodes)")
    print(f"  Episode lengths: {sorted(set(lengths))} (expected {expected_length})")
    if set(lengths) != {expected_length}:
        print("FAILED: unexpected episode length\n")
        sys.exit(1)

    # print the return stats
    print(f"  Return: min {min(returns):.3f} | median {np.median(returns):.3f} | max {max(returns):.3f}\n")
 
    env.close()

def main():
    # Create a parser to check different envs
    parser = argparse.ArgumentParser(description="Validate truss environments.")
    parser.add_argument(
        "--env",
        choices=["parametric", "constructive", "all"],
        default="both",
        help="Which environment to validate (default = all envs)",
    )
    args = parser.parse_args()
 
    if args.env in ("parametric", "all"):
        validate_parametric()
    if args.env in ("constructive", "all"):
        validate_constructive()

if __name__ == "__main__":
    main()