from typing import Union
import numpy as np
import scipy.integrate
import gym
import torch
import cv2

class PendulumEnv_NP(gym.Env):
    def __init__(self, l=1.0, g=9.81, dt=0.02, mode="rk4", discontinuity=False, img_obs=False):
        super().__init__()
        self.l = l
        self.g = g
        self.dt = dt
        self.mode = mode
        self.img_obs = img_obs
        if not img_obs:
            # state is [sin(theta), cos(theta), theta_dot]
            self.observation_space = gym.spaces.Box(np.array([-1., -1., -4*np.pi]), np.array([1., 1., 4*np.pi]))
        else:
            self.img_h, self.img_w = 64, 64
            img_size = (self.img_h, self.img_w, 3)
            img_space = gym.spaces.Box(np.zeros(img_size), np.ones(img_size))
            vel_space = gym.spaces.Box(np.array([-4*np.pi]), np.array([4*np.pi]))
            self.observation_space = gym.spaces.Tuple((vel_space, img_space))
        self.action_space = gym.spaces.Box(np.array([-3.0]), np.array([3.0]))
        self.curr_state = None
        self.has_discontinuity = discontinuity

    def reset(self):
        theta, theta_dot = np.random.uniform([-1,-0.5],[1,0.5])
        self.curr_state = np.array([np.sin(theta), np.cos(theta), theta_dot])
        return self._obs()

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("Only rgb_array mode is supported")
        obs = np.zeros((self.img_h, self.img_w, 3))
        color = (255, 255, 255)
        center = (self.img_w//2, self.img_h//2)
        cv2.circle(obs, center, 4, color, -1)
        pendulum_len = 24 # px
        # rotate 90deg CCW to get rotation in image
        cos_th, sin_th = self.curr_state[0], -self.curr_state[1]
        pendulum_end_px = np.round(np.array([cos_th, sin_th]) * pendulum_len).astype(int)
        x, y = 32 + np.array([1, -1]) * pendulum_end_px
        cv2.circle(obs, (x, y), 4, color, -1)
        cv2.line(obs, center, (x, y), color, 8)
        return obs.astype(np.uint8)

    def _dynamics(self, s, a):
        out = np.array([s[2] * s[1],
                        s[2] * -s[0],
                        -(self.g/self.l)*s[0]+a])
        return out

    def _obs(self):
        if not self.img_obs:
            return self.curr_state
        else:
            obs = self.render() / 255.
            return self.curr_state[-1:], obs

    def set_state(self, state: np.ndarray):
        assert state.shape == (3,)
        self.curr_state = state

    def get_state(self):
        return self.curr_state

    def forward_solver(self, curr_state, a):
        res = scipy.integrate.solve_ivp(lambda _,s: self._dynamics(s, a),
                                                        (0, self.dt),
                                                        curr_state,
                                                        method="RK45",
                                                        dense_output=True)
        return res.sol(self.dt)

    def _reward(self, s, action, next_state):
        theta = np.arctan2(s[0], s[1])
        if theta < 0:
            theta += 2*np.pi
        state = np.array([theta, s[2]])
        goal = np.array([np.pi, 0.0])
        return -0.5*np.linalg.norm(state-goal)**2 - 0.5*action**2

    def step(self, action: np.ndarray):
        a = action.item()
        if self.mode == "forward":
            next_state = self.curr_state + self.dt * self._dynamics(self.curr_state, a)
        else:
            if self.mode == "backward":
                mode = "BDF"
            elif self.mode == "rk4":
                mode = "RK45"
            else:
                raise ValueError(f"Unrecognized mode {self.mode}")
            res = scipy.integrate.solve_ivp(lambda _,s: self._dynamics(s, a),
                                                        (0, self.dt),
                                                        self.curr_state,
                                                        method=mode,
                                                        dense_output=True)
            next_state = res.sol(self.dt)
        rew = self._reward(self.curr_state, action, next_state)
        self.curr_state = next_state

        if self.has_discontinuity:
            theta = np.arctan2(self.curr_state[0], self.curr_state[1])
            if np.abs(theta - np.pi/6) < 1e-2 or np.abs(theta - np.pi/3) < 1e-2 :
                self.curr_state[2] = -self.curr_state[2]

        return self._obs(), np.array(rew), False, {}

class PendulumEnv_Torch(gym.Env):
    def __init__(self, l=1.0, g=9.81, dt=0.02, device="cpu"):
        super().__init__()
        self.l = l
        self.g = g
        self.dt = dt
        self.device = device
        # state is [sin(theta), cos(theta), theta_dot]
        self.observation_space = gym.spaces.Box(np.array([-1., -1., -4*np.pi]),
                                           np.array([1., 1., 4*np.pi]))
        self.action_space = gym.spaces.Box(np.array([-3.0]), np.array([3.0]))
        self.curr_state = None

    def _torchify(self, *xs):
        ret = tuple(torch.as_tensor(x).float().to(self.device) for x in xs)
        return ret[0] if len(ret) == 1 else ret

    def reset(self):
        dist = torch.distributions.Uniform(*self._torchify([-1,-0.5], [1,0.5]))
        theta, theta_dot = dist.sample().to(self.device)
        self.curr_state = torch.stack([torch.sin(theta), torch.cos(theta), theta_dot])
        return self.curr_state

    def _dynamics(self, s, a):
        out = torch.stack([s[2] * s[1],
                        s[2] * -s[0],
                        -(self.g/self.l)*s[0]+a])
        return out

    def set_state(self, state):
        state = self._torchify(state)
        assert state.shape == (3,)
        self.curr_state = state

    def to(self, device):
        self.device = device
        if self.curr_state:
            self.curr_state = self.curr_state.to(device)

    def _reward(self, s, action, next_state):
        theta = torch.arctan2(s[0], s[1])
        if theta < 0:
            theta = theta + 2*np.pi
        state = torch.stack([theta, s[2]])
        goal = torch.tensor([np.pi, 0.0], device=self.device)
        return -0.5*torch.norm(state-goal)**2 - 0.5*action**2

    def step(self, action):
        a = self._torchify(action)
        next_state = self.curr_state + self.dt * self._dynamics(self.curr_state, a.reshape(-1)[0])
        rew = self._reward(self.curr_state, a, next_state)
        self.curr_state = next_state
        return self.curr_state, rew, False, {}

class PendulumEnvCont(gym.Wrapper):
    env: Union[PendulumEnv_NP, PendulumEnv_Torch]
    def __init__(self, l=1.0, g=9.81, dt=0.02, backend="numpy", **kwargs):
        if backend == "numpy":
            env = PendulumEnv_NP(l, g, dt, discontinuity=False, **kwargs)
        elif backend == "torch":
            env = PendulumEnv_Torch(l, g, dt, **kwargs)
        else:
            raise ValueError(f"Unknown backend {backend}")
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def set_state(self, state):
        return self.env.set_state(state)

    def step(self, action):
        return self.env.step(action)

class PendulumEnvContImg(PendulumEnv_NP):
    def __init__(self, l=1.0, g=9.81, dt=0.02):
        super().__init__(l, g, dt, img_obs=True)

class PendulumEnvDiscont(gym.Wrapper):
    env: Union[PendulumEnv_NP, PendulumEnv_Torch]
    def __init__(self, l=1.0, g=9.81, dt=0.02, backend="numpy", **kwargs):
        if backend == "numpy":
            env = PendulumEnv_NP(l, g, dt, discontinuity=True, **kwargs)
        elif backend == "torch":
            env = PendulumEnv_Torch(l, g, dt, **kwargs)
        else:
            raise ValueError(f"Unknown backend {backend}")
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def set_state(self, state):
        return self.env.set_state(state)

    def step(self, action):
        return self.env.step(action)

if __name__ == "__main__":
    # Initialize environment
    env = PendulumEnvContImg()
    obs = env.reset()

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('pendulum_output.mp4', fourcc, 50.0, (64, 64))

    num_steps = int(200)
    for _ in range(num_steps):
        x = env.get_state()
        theta = np.arctan2(x[0], x[1])
        if theta < 0:
            theta += 2*np.pi
        if np.abs(theta - np.pi) < 0.1:
            action = -20.11*(theta - np.pi) - 7.08 * x[2]
        else:
            action = -x[2] * (0.5 * x[2]**2 - 9.81*x[1] - 9.81)
        print(action, end=", ")
        action += np.random.randn() * 5
        print(action)

        obs, _, _, _ = env.step(action)
        img = (obs[1] * 255).astype(np.uint8)  # Convert to uint8 for video
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Write frame to video

    # Release the video writer
    out.release()