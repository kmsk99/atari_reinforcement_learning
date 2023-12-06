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
from collections import namedtuple

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
    return frame.unsqueeze(0)


# 체크포인트 로드 함수
def load_checkpoint(checkpoint_dir=checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")))
    if not checkpoints:
        return None, None  # 체크포인트가 없을 경우 None 반환
    latest_checkpoint = checkpoints[-1]  # 가장 최근의 체크포인트
    checkpoint = torch.load(latest_checkpoint)
    return checkpoint["state"], checkpoint["scores"]


class PolicyNet(nn.Module):
    def __init__(self, num_actions):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self._feature_size(), 512)
        self.fc2 = nn.Linear(512, num_actions)  # 행동 수에 해당하는 출력 크기

    def _feature_size(self):
        return (
            nn.Sequential(self.conv1, self.conv2, self.conv3)
            .forward(torch.zeros(1, 1, 84, 84))
            .view(1, -1)
            .size(1)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # 데이터 평탄화
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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self._feature_size(), 512)
        self.fc2 = nn.Linear(512, 1)  # 출력은 단일 값 (상태의 가치)

    def _feature_size(self):
        return (
            nn.Sequential(self.conv1, self.conv2, self.conv3)
            .forward(torch.zeros(1, 1, 84, 84))
            .view(1, -1)
            .size(1)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # 데이터 평탄화
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# 트라젝토리 수집 함수
def collect_trajectory(env, policy_net):
    Trajectory = namedtuple("Trajectory", "states actions rewards dones logp")
    state_list, action_list, reward_list, dones_list, logp_list = [], [], [], [], []
    state, _ = env.reset()
    done = False
    steps = 0
    state = preprocess(state)

    while not done:
        action, logp = policy_net.get_action_and_logp(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state_list.append(state)
        action_list.append(action)
        reward_list.append(reward)
        dones_list.append(done)
        logp_list.append(logp)
        state = preprocess(next_state)
        steps += 1

    return Trajectory(state_list, action_list, reward_list, dones_list, logp_list)


def calc_returns(rewards):
    dis_rewards = [gamma**i * r for i, r in enumerate(rewards)]
    return [sum(dis_rewards[i:]) for i in range(len(dis_rewards))]


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
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.01)

    print_interval = 20
    score = 0.0
    average_score = 0.0

    scores = []  # 에피소드별 점수 저장 리스트

    # 체크포인트 로드 (필요한 경우)
    checkpoint, loaded_scores = load_checkpoint()
    start_episode = 0  # 시작 에피소드 번호 초기화
    if checkpoint:
        policy_net.load_state_dict(checkpoint["policy_model_state"])
        value_net.load_state_dict(checkpoint["value_model_state"])
        policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state"])
        value_optimizer.load_state_dict(checkpoint["value_optimizer_state"])
        start_episode = checkpoint["episode"]
        scores = loaded_scores  # 이전 점수 이력 로드

    checkpoint_interval = 100  # 체크포인트 저장 간격

    num_iter = 3001
    num_traj = 1

    for n_epi in range(start_episode, num_iter):
        traj_list = [collect_trajectory(env, policy_net) for _ in range(num_traj)]
        returns = [calc_returns(traj.rewards) for traj in traj_list]

        # ====================================#
        # policy gradient with base function #
        # ====================================#
        policy_loss_terms = [
            -1.0 * traj.logp[j] * (returns[i][j] - value_net(traj.states[j].to(device)))
            for i, traj in enumerate(traj_list)
            for j in range(len(traj.actions))
        ]

        # ====================================#
        # policy gradient with reward-to-go  #
        # ====================================#
        # policy_loss_terms = [
        #     -1.0 * traj.logp[j] * (torch.Tensor([returns[i][j]]).to(device))
        #     for i, traj in enumerate(traj_list)
        #     for j in range(len(traj.actions))
        # ]

        # ====================================#
        # policy gradient                    #
        # ====================================#
        # policy_loss_terms = [
        #     -1.0 * traj.logp[j] * (torch.Tensor([returns[i][0]]).to(device))
        #     for i, traj in enumerate(traj_list)
        #     for j in range(len(traj.actions))
        # ]

        policy_loss = 1.0 / num_traj * torch.cat(policy_loss_terms).sum()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_loss_terms = [
            1.0
            / len(traj.actions)
            * (value_net(traj.states[j].to(device)) - returns[i][j]) ** 2.0
            for i, traj in enumerate(traj_list)
            for j in range(len(traj.actions))
        ]
        value_loss = 1.0 / num_traj * torch.cat(value_loss_terms).sum()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        score = sum([traj_returns[0] for traj_returns in returns]) / num_traj
        scores.append(score)

        average_score += score

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "n_episode :{}, score : {:.1f}".format(
                    n_epi, average_score / print_interval
                )
            )
            average_score = 0.0

        # 매 10번째 에피소드마다 그래프 업데이트 및 저장
        if n_epi % 1 == 0:
            plot_scores(scores, "score_plot.png")

        # 체크포인트 저장
        if n_epi % checkpoint_interval == 0 and n_epi != 0:
            save_checkpoint(
                {
                    "policy_model_state": policy_net.state_dict(),
                    "value_model_state": value_net.state_dict(),
                    "policy_optimizer_state": policy_optimizer.state_dict(),
                    "value_optimizer_state": value_optimizer.state_dict(),
                    "episode": n_epi,  # 현재 에피소드 번호 저장
                },
                n_epi,
                scores,
            )

    env.close()


if __name__ == "__main__":
    main()
