# 라이브러리 임포트
import glob
import math
import re
import random
import argparse
import torchvision
import os
import gymnasium as gym
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import WeightedRandomSampler


# 현재 스크립트의 경로를 찾습니다.
current_script_path = os.path.dirname(os.path.realpath(__file__))

# 체크포인트 저장 폴더 경로를 현재 스크립트 위치를 기준으로 설정합니다.
checkpoint_dir = os.path.join(current_script_path, "checkpoints")


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


# 체크포인트 로드 함수
def load_checkpoint(checkpoint_dir=checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")))
    if not checkpoints:
        return None, None  # 체크포인트가 없을 경우 None 반환

    # 파일 이름에서 숫자를 추출하고 정수로 변환하는 람다 함수
    extract_number = lambda filename: int(
        re.search(r"checkpoint_(\d+).pth", filename).group(1)
    )

    # 체크포인트 파일을 숫자 기준으로 정렬 (이제 숫자는 정수로 처리됨)
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")), key=extract_number
    )

    latest_checkpoint = checkpoints[-1]  # 가장 최근의 체크포인트
    checkpoint = torch.load(latest_checkpoint)
    return checkpoint["state"], checkpoint["scores"]


class PrioritizedReplayBuffer:
    def __init__(self, buffer_limit, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.priorities = collections.deque(maxlen=buffer_limit)
        self.device = device
        self.cumulative_reward = 0.01  # 누적 보상 초기화

    def put(self, transition, set_priority=True):
        s, a, r, s_prime, done_mask = transition
        if set_priority == True:
            self.cumulative_reward += r  # 누적 보상 갱신

            if done_mask == 0:  # 에피소드가 끝났다면 누적 보상 초기화
                priority = self.cumulative_reward
                self.cumulative_reward = 0.01  # 누적 보상 초기화
            else:
                priority = self.cumulative_reward

            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer.append(transition)
            self.priorities.append(1.0)  # 우선순위 1.0으로 설정

    def sample(self, n):
        sampler = WeightedRandomSampler(self.priorities, n, replacement=True)
        indices = list(sampler)

        mini_batch = [self.buffer[idx] for idx in indices]
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            s_prime_lst.append(s_prime)
            a_lst.append([a])
            r_lst.append([r])
            done_mask_lst.append([done_mask])

        return (
            torch.stack(s_lst).float().to(self.device),
            torch.tensor(a_lst, dtype=torch.long).to(self.device),
            torch.tensor(r_lst, dtype=torch.float).to(self.device),
            torch.stack(s_prime_lst).float().to(self.device),
            torch.tensor(done_mask_lst, dtype=torch.float).to(self.device),
            indices,  # 반환 값에 indices 추가
        )

    def update_priority(self, idx, priority):
        self.priorities[idx] = priority

    def size(self):
        return len(self.buffer)  # 버퍼 크기 반환


# 그래프 그리기 및 저장 함수 업데이트
def plot_scores(scores, filename):
    plt.figure(figsize=(12, 6))
    # plt.plot(scores, label="Score", color="green")

    # scores 배열이 2000개를 초과하는 경우 첫 2000개 요소로 제한
    # if len(scores) > 2000:
    #     scores = scores[:2000]

    # 10개 이동평균 계산
    moving_avg_10 = [np.mean(scores[max(0, i - 9) : i + 1]) for i in range(len(scores))]
    plt.plot(moving_avg_10, label="10-episode Moving Avg", color="blue")

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


def visualize_filters(model, layer_name, epoch, save_path):
    # 해당 층의 가중치 추출
    for name, layer in model.named_modules():
        if name == layer_name:
            weights = layer.weight.data.cpu()

    # 필터 정규화
    min_val = weights.min()
    max_val = weights.max()
    filters = (weights - min_val) / (max_val - min_val)

    # 필터 시각화 및 저장
    grid_img = torchvision.utils.make_grid(filters, nrow=8, normalize=True, padding=1)

    plt.imshow(grid_img[0:4].permute(1, 2, 0))
    plt.title(f"{layer_name} Filters at Epoch {epoch}")
    plt.axis("off")
    plt.savefig(
        # f"{save_path}\\filters\\{layer_name}_epoch_{epoch}.png", bbox_inches="tight"
        os.path.join(save_path, f"filters/filters_{layer_name}_epoch_{epoch}.png"),
        bbox_inches="tight",
    )
    plt.close()


def visualize_layer_output(model, layer_name, save_path, epoch):
    # 해당 레이어의 출력 추출
    output = model.activations[layer_name].squeeze()

    if output.dim() == 4:
        output = output[0]

    # 첫 번째 채널을 시각화 (다른 채널도 선택 가능)
    img = output.unsqueeze(1).cpu()  # 채널 차원 추가

    grid_img = torchvision.utils.make_grid(img, nrow=8, normalize=True, padding=1)

    plt.imshow(grid_img.permute(1, 2, 0), cmap="gray")
    plt.title(f"Output of {layer_name} at Epoch {epoch}")
    plt.axis("off")
    plt.savefig(
        os.path.join(save_path, f"layers/{layer_name}_output_epoch_{epoch}.png"),
        bbox_inches="tight",
    )
    plt.close()


def preprocess(frame1, frame2=None):
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.CenterCrop((175, 150)),
            T.Resize((84, 84)),
            T.ToTensor(),
        ]
    )
    if frame2 is not None:
        new_frame = 1.667 * (transform(frame2) - 0.4 * transform(frame1))
    else:
        new_frame = transform(frame1)
    return new_frame.squeeze(0)  # 배치 차원 제거
