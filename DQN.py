import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
import os
import collections
import numpy as np
from typing import Tuple, Union
import signal
import cv2
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ReplayBuffer:
    def __init__(self, maxlen: int = 10000) -> None:
        # max length of buffer
        self.maxlen = maxlen
        self.obs_buffer = list()
        # reward, action, done buffer
        self.rad_buffer = list()

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        if len(self.obs_buffer) >= self.maxlen:
            self.obs_buffer.pop(0)
            self.rad_buffer.pop(0)
        self.obs_buffer.append(obs)
        self.rad_buffer.append((reward, action, done))

    def sample(
        self, batch_size: int = 64, history_len: int = 4
    ) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        rand_list, sample_list = [], []
        while True:
            ind = random.randint(0 + history_len - 1, len(self.obs_buffer) - 1 - 1)
            if ind in rand_list or self.rad_buffer[ind][1] == -1:
                continue
            rand_list.append(ind)
            obs = self.obs_buffer[ind - history_len + 1 : ind + 1]
            next_obs = self.obs_buffer[ind - history_len + 2 : ind + 2]
            reward, action, done = self.rad_buffer[ind]
            sample_list.append([obs, action, reward, next_obs, done])
            if len(rand_list) == batch_size:
                break
        obs, action, reward, next_obs, done = zip(*sample_list)
        return (np.array(obs), action, reward, np.array(next_obs), done)

    def size(self) -> int:
        return len(self.obs_buffer)


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
        update_freq: int = 1000,
        update_rate: float = 0.5,
        gamma: float = 0.99,
        lr: float = 1e-5,
        epsilon_init: float = 1,
        epsilon_decay: float = 1e-4,
        epsilon_limit: float = 0.1,
        in_channel: int = 3,
        log_freq: int = 100,
    ) -> None:
        (
            self.action_dim,
            self.gamma,
            self.epsilon,
            self.epsilon_init,
            self.epsilon_limit,
            self.epsilon_decay,
            self.update_freq,
            self.update_rate,
            self.lr,
            self.log_freq,
        ) = (
            action_dim,
            gamma,
            epsilon_init,
            epsilon_init,
            epsilon_limit,
            epsilon_decay,
            update_freq,
            update_rate,
            lr,
            log_freq,
        )
        # q_net and q_target_net
        self.q_net = Net(self.action_dim, in_channel).to(DEVICE)
        self.q_target_net = Net(self.action_dim, in_channel).to(DEVICE)
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optim, T_0=10000, T_mult=2)
        self.episode_num, self.step_num = 0, 0
        self.return_list, self.lr_list = [], []
        self.q_net.train()
        self.q_target_net.eval()

    def select_action(self, obs: np.ndarray) -> int:
        self.step_num += 1
        # add a dim to obs
        x = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(DEVICE)
        # ε-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            # get Q from value_net
            with torch.no_grad():
                self.q_net.eval()
                values = self.q_net(x)
            return int(torch.argmax(values).item())

    def select_best_action(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(DEVICE)
        # select action without ε-greedy policy
        with torch.no_grad():
            self.q_net.eval()
            values = self.q_net(x)
        return int(torch.argmax(values).item())

    def update(self, samples: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> float:
        # unpack samples
        obs, action, reward, next_obs, done = samples
        obs = torch.tensor(obs, dtype=torch.float).to(DEVICE)
        action = torch.tensor(action, dtype=torch.int64).reshape(-1, 1).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float).reshape(-1, 1).to(DEVICE)
        next_obs = torch.tensor(next_obs, dtype=torch.float).to(DEVICE)
        done = torch.tensor(done, dtype=torch.float).reshape(-1, 1).to(DEVICE)

        # get q from q_net
        self.q_net.train()
        q = self.q_net(obs).gather(1, action)
        # get q_target from q_target_net
        q_target = reward + self.gamma * self.q_target_net(next_obs).max(1)[0].view(
            -1, 1
        ) * (1 - done)
        # compute loss according to q and q_target
        loss = F.mse_loss(q, q_target)
        # use gradient descent to optimize nn
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # self.scheduler.step()
        # synchronize q_target_net with q_net
        if self.step_num % self.update_freq == 0:
            # self.target_net.load_state_dict(self.value_net.state_dict())
            for param1, param2 in zip(
                self.q_net.parameters(), self.q_target_net.parameters()
            ):
                new_param_data = (
                    self.update_rate * param1.data
                    + (1 - self.update_rate) * param2.data
                )
                param2.data.copy_(new_param_data)
        # reduce the epsilon
        self.epsilon = max(
            self.epsilon_limit,
            self.epsilon_init
            - (self.epsilon_init - self.epsilon_limit)
            / self.epsilon_decay
            * self.step_num,
        )
        return loss.item()


def plot_return_curve(return_list: list, lr_list: list, step: int):
    plt.figure(figsize=(10, 10))
    # plt.subplot(2, 1, 1)
    plt.xlabel("The Number of Episodes")
    plt.ylabel(f"Last {step} Episodes' Mean Return")
    xdata, ydata = [i * step for i in range(len(return_list))], return_list
    plt.plot(xdata, ydata)
    # plt.subplot(2, 1, 2)
    # plt.xlabel("The Number of Episodes")
    # plt.ylabel(f"Learning Rate")
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # xdata, ydata = [i * step for i in range(len(lr_list))], lr_list
    # plt.plot(xdata, ydata)
    plt.savefig("return_lr_curve.svg")
    plt.close()


def img_preprocess(x: np.ndarray) -> np.ndarray:
    x = cv2.cvtColor(cv2.resize(x, (84, 84)), cv2.COLOR_RGB2GRAY)
    _, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
    return x.astype(np.float64)


def signal_handler(agent: Union[DQN, None]):
    def handler(signum, frame):
        if agent != None:
            print("\n\nPROGRAM IS TERMINATED, CHECKPOINT IS SAVED AS `./checkpoint.pt`")
            plot_return_curve(agent.return_list, agent.lr_list, agent.log_freq)
            torch.save(agent, "checkpoint.pt")
        exit(0)

    return handler


FRAME_SKIP = 4
HISTORY_LEN = 4


def learn(env_id: str = "Breakout-v4", epsilon: float = -1):
    # setup game environment
    env = gym.make(
        env_id,
        obs_type="rgb",
        frameskip=FRAME_SKIP,
        repeat_action_probability=0,
    )
    env.reset()

    # setup parameteres
    EPISODES_NUM = 1_00000
    LR = 1e-4
    # update q_target_net to q_net every UPDATE_FREQ steps
    UPDATE_FREQ = 1_0000
    # q_target_net = UPDATE_RATE * q_net + (1 - UPDATE_RATE) * q_target_net
    UPDATE_RATE = 1
    # do agent.update every TRAIN_FREQ steps
    TRAIN_FREQ = 10
    ACTION_DIM = env.action_space.n  # type: ignore
    BUFFER_MINLEN = 1_0000
    BUFFER_MAXLEN = 2_00000
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPSILON_INIT = 1.0
    # when the num of steps comes to EPSILON_DECAY, agent.epsilon
    EPSILON_DECAY = 1_00_000
    EPSILON_LIMIT = 0.1
    # save model every STORE_INTERVAL steps
    STORE_INTERVAL = 1_000
    # record training info every LOG_FREQ episodes
    LOG_FREQ = 1_00

    # compute mean return
    mean_return = 0
    # use a deque to store frame history
    frame_history = collections.deque(maxlen=HISTORY_LEN)

    # setup replaybuffer and DQN agent
    buffer = ReplayBuffer(BUFFER_MAXLEN)
    agent = None
    if os.path.exists("checkpoint.pt"):
        agent = torch.load("checkpoint.pt")
    else:
        agent = DQN(
            action_dim=ACTION_DIM,
            in_channel=HISTORY_LEN,
            update_freq=UPDATE_FREQ,
            update_rate=UPDATE_RATE,
            gamma=GAMMA,
            epsilon_init=EPSILON_INIT,
            epsilon_decay=EPSILON_DECAY,
            epsilon_limit=EPSILON_LIMIT,
            lr=LR,
            log_freq=LOG_FREQ,
        )
    if epsilon != -1:
        assert 0 <= epsilon <= 1
        agent.epsilon = epsilon
    signal.signal(signal.SIGINT, signal_handler(agent))
    # start training
    for _ in range(EPISODES_NUM):
        # reset
        episode_return = 0
        frame_history.clear()
        obs, _ = env.reset()
        obs = img_preprocess(obs)
        frame_history.append(obs)
        total_loss, count = 0, 0
        while True:
            # combine a set of obs to obs_multi
            obs_multi = [np.zeros(obs.shape)] * (
                HISTORY_LEN - len(frame_history)
            ) + list(frame_history)
            # select an action
            action = agent.select_action(np.array(obs_multi))
            # perform an action and get feedback
            next_obs, reward, done, _, _ = env.step(action)
            next_obs = img_preprocess(next_obs)
            # update the replaybuffer
            buffer.add(obs, action, float(reward), done)
            # update the frame_history and episode_return
            frame_history.append(next_obs)
            episode_return += float(reward)
            obs = next_obs
            # train the q_net and q_target_net
            if buffer.size() >= BUFFER_MINLEN:
                count += 1
                if count % TRAIN_FREQ == 0:
                    samples = buffer.sample(BATCH_SIZE, HISTORY_LEN)
                    loss = agent.update(samples)
                    total_loss += loss
            # judge if the game is over
            if bool(done):
                # add the final obs
                buffer.add(next_obs, -1, -1, True)
                break
        if buffer.size() >= BUFFER_MINLEN:
            agent.episode_num += 1
            mean_return += episode_return / LOG_FREQ
        # print training info on the screen
        terminal_width = os.get_terminal_size().columns
        episode_id = agent.episode_num
        print(
            " " * (terminal_width - 1)
            + "\r {:<4d}: epsilon={:.3f}, step={}, lr={:.2e}, buffer_size={:<5d}, return={:.2f}, loss={:.2e}".format(
                agent.episode_num,
                agent.epsilon,
                agent.step_num,
                agent.scheduler.get_last_lr()[0],
                buffer.size(),
                episode_return,
                total_loss / (count / TRAIN_FREQ) if count != 0 else 0,
            )[
                : terminal_width - 1
            ],
            end="\r",
        )
        if episode_id % LOG_FREQ == 0 and episode_id != 0:
            agent.return_list.append(mean_return)
            agent.lr_list.append(agent.scheduler.get_last_lr()[0])
            plot_return_curve(agent.return_list, agent.lr_list, LOG_FREQ)
            mean_return = 0
        if episode_id % STORE_INTERVAL == 0 and episode_id != 0:
            torch.save(agent, f"./models/DQN-{env_id}-{episode_id}.pt")
    torch.save(agent, f"./DQN-{env_id}.pt")
    plot_return_curve(agent.return_list, agent.lr_list, LOG_FREQ)


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
        episode_return = 0
        while True:
            # select an action
            obs_multi = [np.zeros(obs.shape)] * (
                HISTORY_LEN - len(frame_history)
            ) + list(frame_history)
            action = agent.select_action(np.array(obs_multi))
            # perform an action and get feedback
            next_obs, reward, done, _, _ = env.step(action)
            next_obs = img_preprocess(next_obs)
            episode_return += float(reward)
            # update the obs and final_return
            frame_history.append(next_obs)
            if bool(done):
                break
        print(f"return {episode_return}")


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
    parser.add_argument("-e", "--epsilon", help="set epsilon for training")

    args = parser.parse_args()
    env_id = args.id if args.id is not None else "Breakout-v4"
    epsilon = float(args.epsilon) if args.epsilon is not None else -1
    if args.delete:
        if os.path.exists("./models"):
            shutil.rmtree("./models")
        os.mkdir("./models")
        if os.path.exists("checkpoint.pt"):
            os.remove("checkpoint.pt")
    if not args.play:
        learn(env_id, epsilon)
    else:
        play(
            args.play,
            env_id,
            args.mode if args.mode is not None else "rgb_array",
        )
