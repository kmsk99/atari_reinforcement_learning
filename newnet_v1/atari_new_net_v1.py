# 라이브러리 임포트
import math
import random
import argparse
import os
import gymnasium as gym
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from utils import (
    load_checkpoint,
    plot_scores,
    preprocess,
    save_checkpoint,
    PrioritizedReplayBuffer,
    visualize_filters,
    visualize_layer_output,
    GAMMA,
    MEMORY_SIZE,
    BATCH_SIZE,
    TRAINING_FREQUENCY,
    TARGET_NETWORK_UPDATE_FREQUENCY,
    MODEL_PERSISTENCE_UPDATE_FREQUENCY,
    REPLAY_START_SIZE,
    EXPLORATION_MAX,
    EXPLORATION_MIN,
    EXPLORATION_TEST,
    EXPLORATION_STEPS,
)

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터 설정
gamma = 0.99  # 할인계수
buffer_limit = 50_000  # 버퍼 크기
batch_size = 32  # 배치 크기

# 현재 스크립트의 경로를 찾습니다.
current_script_path = os.path.dirname(os.path.realpath(__file__))

# 체크포인트 저장 폴더 경로를 현재 스크립트 위치를 기준으로 설정합니다.
checkpoint_dir = os.path.join(current_script_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(os.path.join(current_script_path, "filters"), exist_ok=True)
os.makedirs(os.path.join(current_script_path, "layers"), exist_ok=True)


class Qnet(nn.Module):
    def __init__(self, num_actions):
        super(Qnet, self).__init__()
        # 입력 크기: 84x84x1
        # 출력 크기: 32x20x20
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        # 입력 크기: 32x20x20
        # 출력 크기: 64x9x9
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 입력 크기: 64x9x9
        # 출력 크기: 64x7x7
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 출력 특성 크기: 64 * 7 * 7 = 3136
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)

        self.l1 = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            self.flatten
        )

        self.l2 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

    def forward(self, x):
        x = self.l1(x)
        actions = self.l2(x)
        return actions

    def sample_action(self, obs, epsilon):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(device)

        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, out.size(-1) - 1)
        else:
            return out.argmax().item()

    def register_hooks(self, layer_names):
        self.activations = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        for name in layer_names:
            layer = getattr(self, name)
            layer.register_forward_hook(get_activation(name))


# 훈련 함수 정의
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask, indices = memory.sample(BATCH_SIZE)

        s = s.float().to(device)
        s_prime = s_prime.float().to(device)
        r = r.float().to(device)
        done_mask = done_mask.float().to(device)
        a = a.to(device)

        q_out = q(s)
        q_a = q_out.gather(1, a)

        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
        max_q_prime = q_target(s_prime).gather(1, argmax_Q)

        target = r + GAMMA * max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            td_error = torch.abs(q_a - target)

        for idx, error in zip(indices, td_error):
            memory.update_priority(idx, error.item())


# 메인 함수 정의
def main(render):
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array" if render else None)
    q = Qnet(env.action_space.n).to(device)
    q_target = Qnet(env.action_space.n).to(device)
    q_target.load_state_dict(q.state_dict())

    q.register_hooks(["conv1", "conv2", "conv3"])

    frame_queue = deque(maxlen=1)

    # 프레임을 초기화하는 함수
    def init_frames():
        frame_queue.append(torch.zeros(84, 84))

    # memory = ReplayBuffer() 대신에 아래 코드 사용
    memory = PrioritizedReplayBuffer(MEMORY_SIZE, device)

    print_interval = 10
    score = 0.0
    average_score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=0.00025)

    scores = []
    checkpoint, loaded_scores = load_checkpoint()
    start_episode = 0
    if checkpoint:
        q.load_state_dict(checkpoint["model_state"])
        q_target.load_state_dict(checkpoint["target_model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_episode = checkpoint["episode"]
        scores = loaded_scores

    if render:
        for idx in range(10):
            frames = []
            s, _ = env.reset()
            init_frames()
            frame_queue.append(preprocess(s))
            done = False
            plot_scores(scores, "score_plot.png")
            while not done:
                current_state = torch.stack(list(frame_queue), dim=0)
                a = q.sample_action(current_state, EXPLORATION_TEST)
                s_prime, r, terminated, truncated, info = env.step(a)
                frames.append(env.render())
                done = terminated or truncated
                frame_queue.append(preprocess(s_prime, s))
                s = s_prime

                if done:
                    break

            imageio.mimsave(f"gameplay/Breakout_game_play_{idx+1}.gif", frames, fps=30)

    else:
        for n_epi in range(start_episode, 100_001):
            epsilon = max(EXPLORATION_MIN, EXPLORATION_MAX * (1 - n_epi / EXPLORATION_STEPS))
            s, _ = env.reset()
            init_frames()
            frame_queue.append(preprocess(s))
            done = False

            while not done:
                current_state = torch.stack(list(frame_queue), dim=0)
                a = q.sample_action(current_state, epsilon)
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                frame_queue.append(preprocess(s_prime, s))
                next_state = torch.stack(list(frame_queue), dim=0)

                done_mask = 0.0 if done else 1.0
                memory.put((current_state, a, r, next_state, done_mask))

                s = s_prime
                score += r
                average_score += r

                if done:
                    break

            if memory.size() > REPLAY_START_SIZE:
                if n_epi % TRAINING_FREQUENCY == 0:
                    train(q, q_target, memory, optimizer)

            if n_epi % print_interval == 0:
                print(
                    "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                        n_epi,
                        average_score / print_interval,
                        memory.size(),
                        epsilon * 100,
                    )
                )
                average_score = 0.0

            if n_epi % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                q_target.load_state_dict(q.state_dict())

            scores.append(score)
            score = 0

            if n_epi % 100 == 0:
                plot_scores(scores, "score_plot.png")
                visualize_filters(q, "conv1", epoch=n_epi, save_path=current_script_path)
                visualize_filters(q, "conv2", epoch=n_epi, save_path=current_script_path)
                visualize_filters(q, "conv3", epoch=n_epi, save_path=current_script_path)
                visualize_layer_output(q, "conv1", current_script_path, n_epi)
                visualize_layer_output(q, "conv2", current_script_path, n_epi)
                visualize_layer_output(q, "conv3", current_script_path, n_epi)

            if n_epi % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0 and n_epi != 0:
                save_checkpoint(
                    {
                        "model_state": q.state_dict(),
                        "target_model_state": q_target.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "episode": n_epi,
                    },
                    n_epi,
                    scores,
                )

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment to visualize the agent's performance.",
    )
    args = parser.parse_args()

    main(args.render)
