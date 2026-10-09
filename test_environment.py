import sys
from gymnasium.utils.env_checker import check_env
import parameters as default_params
from envs.parametric_truss_env import ParametricTrussEnv


def validate_environment():
    # Initialize the environment
    env = ParametricTrussEnv(params=default_params)

    # Run the Gym Env checker
    try:
        check_env(env.unwrapped, skip_render_check=True)
        print("\nEnvironment PASSED all Gymnasium API compliance checks!\n")
    except Exception as error:
        print(f"\nEnvironment FAILED API checks:\n{error}\n")
        sys.exit(1)

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

if __name__ == "__main__":
    validate_environment()