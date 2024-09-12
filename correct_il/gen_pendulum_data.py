import argparse
from tqdm import tqdm
import numpy as np
import pickle

from envs.pendulum_env import PendulumEnv_NP

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path")
    parser.add_argument("-n", "--n_traj", type=int, default=50)
    parser.add_argument("-H", "--horizon", type=int, default=100)
    return parser.parse_args()

def expert_policy(x):
    theta = np.arctan2(x[0], x[1])
    if theta < 0:
        theta += 2*np.pi
    if np.abs(theta - np.pi) < 0.1:
        action = -20.11*(theta - np.pi) - 7.08 * x[2]
    else:
        action = -x[2] * (0.5 * x[2]**2 - 9.81*x[1] - 9.81)
    return action

def main():
    args = get_args()
    env = PendulumEnv_NP(img_obs=True)

    trajs = []
    total_rews = []
    for _ in tqdm(range(args.n_traj)):
        done = False
        t = 0
        obs = env.reset()
        traj = {"observations": [], "img_observations": [], "actions": []}
        total_rew = 0
        while not done and t < args.horizon:
            state = env.get_state()
            action = expert_policy(state)
            action += np.random.randn() * 5
            next_obs, rew, done, _ = env.step(action)
            total_rew += rew
            traj["observations"].append(obs[0])
            traj["img_observations"].append(obs[1])
            traj["actions"].append(action)
            obs = next_obs
            t += 1
        for k, v in traj.items():
            traj[k] = np.array(v)
        trajs.append(traj)
        total_rews.append(total_rew)

    print(f"Mean return: {np.mean(total_rews)}")
    with open(args.save_path, "wb") as f:
        pickle.dump(trajs, f)


if __name__ == "__main__":
    main()
