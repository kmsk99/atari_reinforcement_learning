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

# 하이퍼파라미터 설정
GAMMA = 0.99

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
    # 파일 이름에서 에피소드 번호 추출
    episode_number = extract_number(latest_checkpoint)
    
    checkpoint = torch.load(latest_checkpoint)
    # 체크포인트에 에피소드 번호 추가
    if "state" in checkpoint and isinstance(checkpoint["state"], dict):
        checkpoint["state"]["episode"] = episode_number
    
    return checkpoint["state"], checkpoint["scores"]


# 그래프 그리기 및 저장 함수 업데이트
def plot_scores(scores, filename):
    plt.figure(figsize=(12, 6))

    # 10개 이동평균 계산
    moving_avg_10 = [np.mean(scores[max(0, i - 9) : i + 1]) for i in range(len(scores))]
    plt.plot(moving_avg_10, label="10-episode Moving Avg", color="blue")

    # 100개 이동평균 계산
    moving_avg_100 = [
        np.mean(scores[max(0, i - 99) : i + 1]) for i in range(len(scores))
    ]
    plt.plot(moving_avg_100, label="100-episode Moving Avg", color="yellow")

    # 1000개 이동평균 계산
    moving_avg_1000 = [
        np.mean(scores[max(0, i - 999) : i + 1]) for i in range(len(scores))
    ]
    plt.plot(moving_avg_1000, label="1000-episode Moving Avg", color="red")

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
            
    # float32 타입으로 명시적 변환 (혼합 정밀도 학습으로 인한 float16 문제 해결)
    weights = weights.float()

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
        os.path.join(save_path, f"filters/filters_{layer_name}_epoch_{epoch}.png"),
        bbox_inches="tight",
    )
    plt.close()


def visualize_layer_output(model, layer_name, save_path, epoch):
    # 해당 레이어의 출력 추출
    output = model.activations[layer_name].squeeze()
    
    # float32 타입으로 명시적 변환 (혼합 정밀도 학습으로 인한 float16 문제 해결)
    output = output.float()

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


def preprocess(frame):
    """
    프레임을 전처리합니다.
    
    Args:
        frame: 원본 RGB 프레임
        
    Returns:
        전처리된 그레이스케일 프레임 (84x84)
    """
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.CenterCrop((175, 150)),
            T.Resize((84, 84)),
            T.ToTensor(),
        ]
    )
    
    return transform(frame).squeeze(0)  # 배치 차원 제거
