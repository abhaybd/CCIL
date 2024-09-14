import argparse
from diffusion_policy.policy import DiffusionPolicy
from envs import *
import d4rl as _
import gym
import os
import cv2
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_path", help="Path to policy file")
    return parser.parse_args()

def main():
    args = get_args()
    policy = DiffusionPolicy.load(args.policy_path)

    env = gym.make("PendulumSwingupContImg-v0")
    env.seed(0)

    video_dir = "diffusion_vids"
    os.makedirs(video_dir, exist_ok=True)

    rews = []
    for trial in tqdm(range(100)):
        policy.clear_obs()
        obs = env.reset()
        done = False
        rew = 0
        video = [env.render(mode="rgb_array")] if video_dir and trial % 10 == 0 else None
        step = 0
        pred_actions = None
        while not done:
            vec, img = obs
            policy.add_obs({"img_observations": img, "state": vec})
            if pred_actions is None:
                pred_actions = policy.get_action()
            action = pred_actions[0]
            pred_actions = pred_actions[1:]
            obs, r, done, _ = env.step(action)
            if video is not None:
                video.append(env.render(mode="rgb_array"))
            rew += r
            step += 1
            if step % 5 == 0:
                pred_actions = None
        rews.append(rew)
        print(f"Trial {trial}: {rew}")

        if video is not None:
            out = cv2.VideoWriter(f"{video_dir}/trial_{trial}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 50, (video[0].shape[1], video[0].shape[0]))
            for frame in video:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()


if __name__ == "__main__":
    main()
