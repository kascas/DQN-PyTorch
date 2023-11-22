import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import random
import os
import collections
import numpy as np
from typing import Tuple
import signal
import cv2

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ReplayBuffer:
    def __init__(self, maxlen: int = 10000) -> None:
        # max length of buffer
        self.maxlen = maxlen
        self.buffer = collections.deque(maxlen=maxlen)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(
        self, batch_size: int = 64
    ) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        obs, action, reward, next_obs, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (np.array(obs), action, reward, np.array(next_obs), done)

    def size(self) -> int:
        return len(self.buffer)


class Net(nn.Module):
    def __init__(self, action_dim: int, in_channel: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x: torch.Tensor):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN:
    def __init__(
        self,
        action_dim: int,
        update_period: int = 1000,
        update_rate: float = 0.5,
        gamma: float = 0.99,
        lr: float = 1e-5,
        epsilon_init: float = 1,
        epsilon_decay: float = 1e-4,
        epsilon_limit: float = 0.1,
        in_channel: int = 3,
    ) -> None:
        (
            self.action_dim,
            self.gamma,
            self.epsilon,
            self.epsilon_limit,
            self.epsilon_decay,
            self.update_period,
            self.update_rate,
            self.lr,
        ) = (
            action_dim,
            gamma,
            epsilon_init,
            epsilon_limit,
            epsilon_decay,
            update_period,
            update_rate,
            lr,
        )
        # q_net and q_target_net
        self.value_net = Net(self.action_dim, in_channel).to(DEVICE)
        self.target_net = Net(self.action_dim, in_channel).to(DEVICE)
        self.value_net.train()
        self.target_net.eval()
        self.optim = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.count = 0
        signal.signal(signal.SIGINT, self.signal_handler)

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_lr(self):
        return self.lr

    def get_count(self):
        return self.count

    def select_action(self, obs: np.ndarray) -> int:
        # add a dim to obs
        x = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(DEVICE)
        # ε-greedy policy
        if random.random() < self.get_epsilon():
            return random.randint(0, self.action_dim - 1)
        else:
            # get Q from target_net
            with torch.no_grad():
                self.value_net.eval()
                values = self.value_net(x)
            return int(torch.argmax(values).item())

    def select_best_action(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(DEVICE)
        # select action without ε-greedy policy
        with torch.no_grad():
            self.value_net.eval()
            values = self.value_net(x)
        return int(torch.argmax(values).item())

    def update(self, samples: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> float:
        # unpack samples
        obs, action, reward, next_obs, done = samples
        obs = torch.tensor(obs, dtype=torch.float).to(DEVICE)
        action = torch.tensor(action, dtype=torch.int64).reshape(-1, 1).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float).reshape(-1, 1).to(DEVICE)
        next_obs = torch.tensor(next_obs, dtype=torch.float).to(DEVICE)
        done = torch.tensor(done, dtype=torch.float).reshape(-1, 1).to(DEVICE)

        # get qsa from q_net
        self.value_net.train()
        qsa = self.value_net(obs).gather(1, action)
        # get qsa_m from q_target_net
        qsa_m = reward + self.gamma * self.target_net(next_obs).max(1)[0].view(
            -1, 1
        ) * (1 - done)
        # compute loss according to qsa and qsa_m
        loss = F.mse_loss(qsa, qsa_m)
        # use gradient descent to optimize nn
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # synchronize q_target_net with q_net
        if self.count % self.update_period == 0:
            # self.target_net.load_state_dict(self.value_net.state_dict())
            for param1, param2 in zip(
                self.value_net.parameters(), self.target_net.parameters()
            ):
                new_param_data = (
                    self.update_rate * param1.data
                    + (1 - self.update_rate) * param2.data
                )
                param2.data.copy_(new_param_data)
        self.count += 1
        # reduce the epsilon
        self.epsilon = max(self.epsilon_limit, self.epsilon - self.epsilon_decay)
        return loss.item()

    def signal_handler(self, signal, frame):
        print(
            "\n=== PROGRAM IS TERMINATED, CHECKPOINT IS SAVED AS `./checkpoint.pt` ==="
        )
        torch.save(self, "checkpoint.pt")
        exit(0)


def img_preprocess(x: np.ndarray) -> np.ndarray:
    x = cv2.cvtColor(cv2.resize(x, (84, 84)), cv2.COLOR_RGB2GRAY)
    _, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
    return x.astype(np.float64)


FRAME_SKIP = 4
HISTORY_LEN = 4


def learn(env_id: str = "Breakout-v4"):
    # setup game environment
    env = gym.make(
        env_id,
        obs_type="rgb",
        frameskip=FRAME_SKIP,
        repeat_action_probability=0,
    )
    env.reset()

    # setup parameteres
    EPISODES_NUM = 100000
    LR = 1e-6
    UPDATE_PERIOD = 200
    UPDATE_RATE = 0.8
    ACTION_DIM = env.action_space.n  # type: ignore
    BUFFER_MINLEN = 10000
    BUFFER_MAXLEN = 40000
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPSILON_INIT = 1.0
    EPSILON_DECAY = 1e-4
    EPSILON_LIMIT = 0.1
    STORE_INTERVAL = 1000

    # record every return during training
    return_list = []
    total_return = 0
    # use a deque to store frame history
    frame_history = collections.deque(maxlen=HISTORY_LEN)

    # setup replaybuffer and DQN agent
    buffer = ReplayBuffer(BUFFER_MAXLEN)
    agent = None
    if os.path.exists("./DQN-agent.pt"):
        agent = torch.load("./DQN-agent.pt")
    else:
        agent = DQN(
            action_dim=ACTION_DIM,
            in_channel=HISTORY_LEN,
            update_period=UPDATE_PERIOD,
            update_rate=UPDATE_RATE,
            gamma=GAMMA,
            epsilon_init=EPSILON_INIT,
            epsilon_decay=EPSILON_DECAY,
            epsilon_limit=EPSILON_LIMIT,
            lr=LR,
        )

    # start training
    episode_id = agent.get_count()
    step_num = 0
    for _ in range(EPISODES_NUM):
        # reset
        final_return = 0
        frame_history.clear()
        obs, _ = env.reset()
        obs = img_preprocess(obs)
        frame_history.append(obs)
        while True:
            step_num += 1
            # select an action
            obs_multi = [np.zeros(obs.shape)] * (
                HISTORY_LEN - len(frame_history)
            ) + list(frame_history)
            action = agent.select_action(np.array(obs_multi))
            # perform an action and get feedback
            next_obs, reward, done, _, _ = env.step(action)
            next_obs = img_preprocess(next_obs)
            # update the replaybuffer
            next_obs_multi = obs_multi[1:] + [next_obs]
            buffer.add(
                np.array(obs_multi),
                action,
                reward,
                np.array(next_obs_multi),
                done,
            )
            # update the obs and final_return
            frame_history.append(next_obs)
            final_return += reward
            # judge if the game is over
            if bool(done):
                break
        return_list.append(final_return)
        total_return += final_return
        total_loss = 0
        # train the q_net and q_target_net
        if buffer.size() >= BUFFER_MINLEN:
            samples = buffer.sample(BATCH_SIZE)
            loss = agent.update(samples)
            total_loss += loss
            episode_id += 1
        # print training info on the screen
        terminal_width = os.get_terminal_size().columns
        print(
            " " * (terminal_width - 1)
            + "\r {:<4d}: epsilon={:.3f}, step={}, lr={:.2e}, buffer_size={:<5d}, return={:.2f}, loss={:.2e}".format(
                episode_id,
                agent.get_epsilon(),
                step_num,
                agent.get_lr(),
                buffer.size(),
                final_return,
                total_loss,
            )[
                : terminal_width - 1
            ],
            end="\r",
        )
        if episode_id % UPDATE_PERIOD == 0 and episode_id != 0:
            print(
                " " * (terminal_width - 1)
                + "\r{}-{} episodes' mean return: {:.3f}".format(
                    episode_id - UPDATE_PERIOD,
                    episode_id,
                    total_return / UPDATE_PERIOD,
                )
            )
            total_return = 0
        if episode_id % STORE_INTERVAL == 0 and episode_id != 0:
            torch.save(agent, f"./models/DQN-{episode_id}.pt")
    torch.save(agent, "./DQN-agent.pt")
    torch.save(return_list, "./return_list.pt")


def play(filepath, env_id: str = "Breakout-v4", render_mode: str = "rgb_array"):
    env = gym.make(
        env_id,
        render_mode=render_mode,
        obs_type="rgb",
        frameskip=FRAME_SKIP,
        repeat_action_probability=0,
    )
    # use a deque to store frame history
    frame_history = collections.deque(maxlen=HISTORY_LEN)

    agent = torch.load(filepath)
    while True:
        # reset
        obs, _ = env.reset()
        obs = img_preprocess(obs)
        frame_history.clear()
        frame_history.append(obs)
        step_num = 0
        final_return = 0
        while True:
            step_num += 1
            # select an action
            obs_multi = [np.zeros(obs.shape)] * (
                HISTORY_LEN - len(frame_history)
            ) + list(frame_history)

            action = agent.select_action(np.array(obs_multi))
            # perform an action and get feedback
            next_obs, reward, done, _, _ = env.step(action)
            next_obs = img_preprocess(next_obs)
            final_return += reward
            # update the obs and final_return
            frame_history.append(next_obs)
            if bool(done):
                break
        print(f"return {final_return}")


if __name__ == "__main__":
    import argparse
    import shutil
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--play", help="use current agent to play the game")
    parser.add_argument(
        "-l", "--learn", help="let agent learn the game", action="store_true"
    )
    parser.add_argument("-d", "--delete", help="delete tmp files", action="store_true")
    parser.add_argument("-i", "--id", help="select atari env")
    parser.add_argument("-m", "--mode", help="select render mode")

    args = parser.parse_args()
    if args.delete:
        if os.path.exists("./models"):
            shutil.rmtree("./models")
        os.mkdir("./models")
        if os.path.exists("./DQN-agent.pt"):
            os.remove("./DQN-agent.pt")
    if not args.play:
        learn(args.id if args.id is not None else "Breakout-v4")
    else:
        play(
            args.play,
            args.id if args.id is not None else "Breakout-v4",
            args.mode if args.mode is not None else "rgb_array",
        )
