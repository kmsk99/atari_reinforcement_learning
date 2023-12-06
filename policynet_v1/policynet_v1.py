# 라이브러리 임포트
import glob
import re
import random
import os
import gymnasium as gym
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import WeightedRandomSampler

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터 설정
gamma = 0.998  # 할인계수
buffer_limit = 100_000  # 버퍼 크기
batch_size = 32  # 배치 크기

# 현재 스크립트의 경로를 찾습니다.
current_script_path = os.path.dirname(os.path.realpath(__file__))

# 체크포인트 저장 폴더 경로를 현재 스크립트 위치를 기준으로 설정합니다.
checkpoint_dir = os.path.join(current_script_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)


def save_checkpoint(
    state, episode, scores, checkpoint_dir=checkpoint_dir, keep_last=10
):
    filepath = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")
    torch.save({"state": state, "scores": scores}, filepath)

    # 파일 이름에서 숫자를 추출하고 정수로 변환하는 람다 함수
    extract_number = lambda filename: int(
        re.search(r"checkpoint_(\d+).pth", filename).group(1)
    )

    # 체크포인트 파일을 숫자 기준으로 정렬 (이제 숫자는 정수로 처리됨)
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")), key=extract_number
    )

    # 오래된 체크포인트 삭제
    while len(checkpoints) > keep_last:
        os.remove(checkpoints.pop(0))  # 가장 오래된 체크포인트 삭제


# 그래프 그리기 및 저장 함수 업데이트
def plot_scores(scores, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label="Score", color="blue")

    # 10개 이동평균 계산
    moving_avg_10 = [np.mean(scores[max(0, i - 9) : i + 1]) for i in range(len(scores))]
    plt.plot(moving_avg_10, label="10-episode Moving Avg", color="green")

    # 50개 이동평균 계산
    moving_avg_50 = [
        np.mean(scores[max(0, i - 49) : i + 1]) for i in range(len(scores))
    ]
    plt.plot(moving_avg_50, label="50-episode Moving Avg", color="red")

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Scores per Episode")
    plt.legend()
    plt.savefig(os.path.join(current_script_path, filename))
    plt.close()


transform = T.Compose(
    [
        T.ToPILImage(),
        T.Grayscale(num_output_channels=1),
        T.Resize((84, 84)),
        T.ToTensor(),
    ]
)


def preprocess(frame):
    frame = transform(frame)
    return frame.squeeze(0)  # 배치 차원 제거


# 이전 프레임을 저장하기 위한 큐
from collections import deque

frame_queue = deque(maxlen=4)


# 프레임을 초기화하는 함수
def init_frames():
    for _ in range(4):
        frame_queue.append(torch.zeros(84, 84))


# 체크포인트 로드 함수
def load_checkpoint(checkpoint_dir=checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")))
    if not checkpoints:
        return None, None  # 체크포인트가 없을 경우 None 반환
    latest_checkpoint = checkpoints[-1]  # 가장 최근의 체크포인트
    checkpoint = torch.load(latest_checkpoint)
    return checkpoint["state"], checkpoint["scores"]


class PrioritizedReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.priorities = collections.deque(maxlen=buffer_limit)

    def put(self, transition, priority):
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, n):
        sampler = WeightedRandomSampler(self.priorities, n, replacement=True)
        indices = list(sampler)

        mini_batch = [self.buffer[idx] for idx in indices]
        s_lst, a_lst, r_lst, logp_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, logp, done_mask = transition
            s_lst.append(s)
            logp_lst.append(logp)
            a_lst.append([a])
            r_lst.append([r])
            done_mask_lst.append([done_mask])

        return (
            torch.stack(s_lst).float().to(device),
            torch.tensor(a_lst, dtype=torch.long).to(device),
            torch.tensor(r_lst, dtype=torch.float).to(device),
            torch.tensor(logp_lst, dtype=torch.float).to(device),
            torch.tensor(done_mask_lst, dtype=torch.float).to(device),
            indices,  # 반환 값에 indices 추가
        )

    def update_priority(self, idx, priority):
        self.priorities[idx] = priority

    def size(self):
        return len(self.buffer)  # 버퍼 크기 반환


class PolicyNet(nn.Module):
    def __init__(self, num_actions):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self._feature_size(), 512)
        self.fc2 = nn.Linear(512, num_actions)  # 행동 수에 해당하는 출력 크기

    def _feature_size(self):
        return (
            nn.Sequential(self.conv1, self.conv2, self.conv3)
            .forward(torch.zeros(1, 4, 84, 84))
            .view(1, -1)
            .size(1)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)  # 소프트맥스 적용

    def get_action_and_logp(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(device)

        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        action_probs = self.forward(obs)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        logp = m.log_prob(action)
        return action.item(), logp

    def act(self, x):
        action, _ = self.get_action_and_logp(x)
        return action


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self._feature_size(), 512)
        self.fc2 = nn.Linear(512, 1)  # 출력은 단일 값 (상태의 가치)

    def _feature_size(self):
        return (
            nn.Sequential(self.conv1, self.conv2, self.conv3)
            .forward(torch.zeros(1, 4, 84, 84))
            .view(1, -1)
            .size(1)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# 훈련 함수 정의
def train(value_net, optimizer_policy, optimizer_value, memory):
    for i in range(10):
        s, a, r, logp, done_mask, indices = memory.sample(batch_size)

        # GPU로 이동
        s = s.float().to(device)
        logp = logp.float().to(device)
        r = r.float().to(device)
        done_mask = done_mask.float().to(device)
        a = a.to(device)

        # 현재 가치와 타겟 가치 계산
        current_values = value_net(s).squeeze(1)  # Squeeze를 사용하여 차원 축소
        target_values = r + gamma * current_values * done_mask

        # 가치 손실 계산
        value_loss = F.mse_loss(current_values, target_values.detach())

        # 정책 손실 계산
        advantage = target_values.detach() - current_values
        policy_loss = -(logp * advantage).mean()

        # 역전파 및 최적화
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        # TD 오류 계산
        with torch.no_grad():
            td_error = torch.abs(target_values - current_values)

        # 우선순위 업데이트
        for idx, error in zip(indices, td_error):
            memory.update_priority(idx, error.item())


# 메인 함수 정의
def main():
    env = gym.make(
        "ALE/SpaceInvaders-v5"
        #    , render_mode="human"
    )  # 환경 초기화
    # 모델을 GPU로 이동
    policy_net = PolicyNet(6).to(device)
    value_net = ValueNet().to(device)

    # 최적화 함수 설정
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.01)
    optimizer_value = optim.Adam(value_net.parameters(), lr=0.01)

    # 메모리 버퍼 초기화
    memory = PrioritizedReplayBuffer()

    print_interval = 20
    score = 0.0
    average_score = 0.0

    # CosineAnnealingLR 스케줄러 초기화
    # scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_policy,
    #     T_max=1000,
    #     eta_min=0.01,
    # )

    scores = []  # 에피소드별 점수 저장 리스트

    # 체크포인트 로드 (필요한 경우)
    checkpoint, loaded_scores = load_checkpoint()
    start_episode = 0  # 시작 에피소드 번호 초기화
    if checkpoint:
        policy_net.load_state_dict(checkpoint["policy_model_state"])
        value_net.load_state_dict(checkpoint["value_model_state"])
        optimizer_policy.load_state_dict(checkpoint["optimizer_policy_state"])
        optimizer_value.load_state_dict(checkpoint["optimizer_value_state"])
        start_episode = checkpoint["episode"]
        scores = loaded_scores  # 이전 점수 이력 로드

    checkpoint_interval = 100  # 체크포인트 저장 간격

    for n_epi in range(start_episode, 3001):
        s, _ = env.reset()
        init_frames()  # 프레임 큐 초기화
        frame_queue.append(preprocess(s))  # 첫 프레임 추가
        done = False

        while not done:
            # 현재 상태를 4개 프레임으로 구성
            current_state = torch.stack(list(frame_queue), dim=0)

            a, logp = policy_net.get_action_and_logp(current_state)
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            done_mask = 0.0 if done else 1.0

            # 이전 코드의 memory.put(...) 대신에 아래 코드 사용
            # 초기 우선순위를 지정합니다. 예를 들어, 일정한 값으로 시작할 수 있습니다.
            initial_priority = 1.0
            memory.put((current_state, a, r / 100.0, logp, done_mask), initial_priority)

            s = s_prime

            score += r
            average_score += r
            if done:
                break

        if memory.size() > 1000:
            train(value_net, optimizer_policy, optimizer_value, memory)
            # scheduler.step()  # 에피소드마다 학습률 업데이트

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}".format(
                    n_epi, average_score / print_interval, memory.size()
                )
            )
            average_score = 0.0

        scores.append(score)  # 점수 저장
        score = 0

        # 매 10번째 에피소드마다 그래프 업데이트 및 저장
        if n_epi % 10 == 0:
            plot_scores(scores, "score_plot.png")

        # 체크포인트 저장
        if n_epi % checkpoint_interval == 0 and n_epi != 0:
            save_checkpoint(
                {
                    "policy_model_state": policy_net.state_dict(),
                    "value_model_state": value_net.state_dict(),
                    "optimizer_policy_state": optimizer_policy.state_dict(),
                    "optimizer_value_state": optimizer_value.state_dict(),
                    "episode": n_epi,  # 현재 에피소드 번호 저장
                },
                n_epi,
                scores,
            )

    env.close()


if __name__ == "__main__":
    main()
