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

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터 설정
gamma = 0.98  # 할인계수
buffer_limit = 50000  # 버퍼 크기
batch_size = 32  # 배치 크기

# 현재 스크립트의 경로를 찾습니다.
current_script_path = os.path.dirname(os.path.realpath(__file__))

# 체크포인트 저장 폴더 경로를 현재 스크립트 위치를 기준으로 설정합니다.
checkpoint_dir = os.path.join(current_script_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)


def save_checkpoint(state, episode, scores, checkpoint_dir=checkpoint_dir, keep_last=2):
    filepath = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")
    torch.save({"state": state, "scores": scores}, filepath)

    # 파일 이름에서 숫자를 추출하는 람다 함수
    extract_number = lambda filename: int(re.search(r"\d+", filename).group())

    # 체크포인트 파일을 숫자 기준으로 정렬
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")), key=extract_number
    )

    # 오래된 체크포인트 삭제
    if len(checkpoints) > keep_last:
        os.remove(checkpoints[0])  # 가장 오래된 체크포인트 삭제


# 체크포인트 로드 함수
def load_checkpoint(checkpoint_dir=checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")))
    if not checkpoints:
        return None, None  # 체크포인트가 없을 경우 None 반환
    latest_checkpoint = checkpoints[-1]  # 가장 최근의 체크포인트
    checkpoint = torch.load(latest_checkpoint)
    return checkpoint["state"], checkpoint["scores"]


# 경험 리플레이 버퍼 정의
class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  # 더블 엔드 큐를 사용한 버퍼 초기화

    def put(self, transition):
        self.buffer.append(transition)  # 버퍼에 경험 추가

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)  # 버퍼에서 무작위로 n개 샘플 추출
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s.transpose(2, 0, 1))  # 차원 순서 변경
            s_prime_lst.append(s_prime.transpose(2, 0, 1))
            a_lst.append([a])
            r_lst.append([r])
            done_mask_lst.append([done_mask])

        # numpy 배열로 변환한 후 torch 텐서로 변환
        return (
            torch.tensor(np.array(s_lst), dtype=torch.float),
            torch.tensor(np.array(a_lst), dtype=torch.long),
            torch.tensor(np.array(r_lst), dtype=torch.float),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float),
            torch.tensor(np.array(done_mask_lst)),
        )

    def size(self):
        return len(self.buffer)  # 버퍼 크기 반환


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.model = nn.Sequential(
            DepthwiseSeparableConv(3, 16, 2),
            SELayer(16),
            DepthwiseSeparableConv(16, 32, 2),
            SELayer(32),
            DepthwiseSeparableConv(32, 64, 2),
            SELayer(64),
            # 추가 컨볼루션 및 SELayer 레이어를 추가할 수 있음
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 6),  # 액션의 개수에 따라 출력 크기 설정
        )

    def _feature_size(self):
        # CNN 출력 크기 계산을 위한 임시 데이터 처리
        with torch.no_grad():
            return (
                nn.Sequential(self.conv1, self.conv2, self.conv3)
                .forward(torch.zeros(1, 3, 210, 160))
                .view(1, -1)
                .size(1)
            )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

    def sample_action(self, obs, epsilon):
        # 행동 선택 메서드
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()  # NumPy 배열을 Tensor로 변환
        obs = obs.to(device).unsqueeze(0)  # 단일 관찰에 배치 차원 추가 및 GPU로 이동
        obs = obs.permute(0, 3, 1, 2)  # 차원 재정렬
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 5)  # 무작위 액션 선택
        else:
            return out.argmax().item()  # 최적의 액션 선택


# 훈련 함수 정의
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # GPU로 이동
        s = s.float().to(device)
        s_prime = s_prime.float().to(device)
        r = r.float().to(device)
        done_mask = done_mask.float().to(device)
        a = a.to(device)

        # 텐서 타입을 Float로 설정
        s = s.float()
        s_prime = s_prime.float()
        r = r.float()
        done_mask = done_mask.float()

        q_out = q(s)
        q_a = q_out.gather(1, a)

        # Double DQN 구현
        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
        max_q_prime = q_target(s_prime).gather(1, argmax_Q)

        target = r + gamma * max_q_prime * done_mask

        # 손실 함수 계산 및 역전파
        loss = F.mse_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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


import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


# 메인 함수 정의
def main():
    env = gym.make(
        "ALE/SpaceInvaders-v5"
        #    , render_mode="human"
    )  # 환경 초기화
    # 모델을 GPU로 이동
    q = Qnet().to(device)
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 10
    score = 0.0
    average_score = 0.0

    # 최적화 함수 설정
    optimizer = optim.Adam(q.parameters(), lr=0)  # 초기 학습률 설정

    # CosineAnnealingLR 스케줄러 초기화
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=50, T_mult=2, eta_max=0.1, T_up=10, gamma=0.5
    )

    scores = []  # 에피소드별 점수 저장 리스트

    # 체크포인트 로드 (필요한 경우)
    checkpoint, loaded_scores = load_checkpoint()
    start_episode = 0  # 시작 에피소드 번호 초기화
    if checkpoint:
        q.load_state_dict(checkpoint["model_state"])
        q_target.load_state_dict(checkpoint["target_model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_episode = checkpoint["episode"]
        scores = loaded_scores  # 이전 점수 이력 로드

    checkpoint_interval = 100  # 체크포인트 저장 간격

    for n_epi in range(start_episode, 3001):
        epsilon = max(0.005, 0.08 - 0.01 * (n_epi / 200))  # 탐험률 조정
        s, _ = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            average_score += r
            if done:
                break

        if memory.size() > 1000:
            train(q, q_target, memory, optimizer)
            scheduler.step()  # 에피소드마다 학습률 업데이트

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, average_score / print_interval, memory.size(), epsilon * 100
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
                    "model_state": q.state_dict(),
                    "target_model_state": q_target.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "episode": n_epi,  # 현재 에피소드 번호 저장
                },
                n_epi,
                scores,
            )

    env.close()


if __name__ == "__main__":
    main()