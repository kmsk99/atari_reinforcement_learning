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
import csv
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from datetime import datetime
import pytz

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
def plot_scores(scores, filename, save_csv=True):
    # CSV 파일 저장
    if save_csv:
        csv_filename = os.path.join(current_script_path, 'scores.csv')
        
        # 데이터 준비 - 점수만 저장
        data = {
            'episode': list(range(1, len(scores) + 1)),
            'score': scores,
        }
        
        # 데이터프레임 생성 및 저장
        df = pd.DataFrame(data)
        df.to_csv(csv_filename, index=False)
        print(f"점수 데이터가 {csv_filename}에 저장되었습니다.")
    
    # CSV 파일에서 데이터 읽기
    csv_filename = os.path.join(current_script_path, 'scores.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        scores = df['score'].values
    
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
    
    # 최고 점수 추적 및 플롯
    max_scores = [max(scores[:i+1]) for i in range(len(scores))]
    plt.plot(max_scores, label="Best Score", color="green", linestyle="--")
    
    # 100개 윈도우의 Moving Best 계산
    window_size = 100
    moving_best = []
    for i in range(len(scores)):
        window_start = max(0, i - window_size + 1)
        moving_best.append(max(scores[window_start:i+1]))
    plt.plot(moving_best, label="Moving Best (window=100)", color="purple", linestyle="-.")
    
    # 100개 윈도우의 Moving Worst 계산
    moving_worst = []
    for i in range(len(scores)):
        window_start = max(0, i - window_size + 1)
        moving_worst.append(min(scores[window_start:i+1]))
    plt.plot(moving_worst, label="Moving Worst (window=100)", color="orange", linestyle="-.")

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
        # f"{save_path}\\filters\\{layer_name}_epoch_{epoch}.png", bbox_inches="tight"
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

# 게임 플레이를 GIF로 저장하는 함수
def save_gameplay_gif(model, env_id, episode_num, max_steps=1000, device="cuda", save_path=None):
    """
    현재 학습된 모델을 사용하여 게임 플레이를 GIF로 저장합니다.
    
    Args:
        model: 현재 학습된 PPO 모델
        env_id: 환경 ID (예: "ALE/Breakout-v5")
        episode_num: 현재 에피소드 번호
        max_steps: 최대 스텝 수
        device: 모델을 실행할 장치
        save_path: GIF를 저장할 경로
    """
    if save_path is None:
        save_path = os.path.join(current_script_path, "gameplay")
    
    # 환경 생성
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStack(env, 4)
    
    frames = []
    obs, _ = env.reset()
    
    done = False
    total_reward = 0
    
    # 최대 스텝 수 또는 게임 종료까지 진행
    for _ in range(max_steps):
        if done:
            break
            
        # 상태를 PyTorch 텐서로 변환
        obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)
        
        # 행동 선택
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs_tensor)
            action = action.cpu().item()
        
        next_obs, r, terminated, truncated, info = env.step(action)
        
        # 렌더링된 프레임 저장
        frames.append(env.render())
        
        done = terminated or truncated
        total_reward += r
        
        # 다음 상태로 업데이트
        obs = next_obs
    
    env.close()
    
    # GIF 저장
    import imageio
    filename = os.path.join(save_path, f"gameplay_episode_{episode_num}.gif")
    imageio.mimsave(filename, frames, fps=30)
    
    return total_reward

# 한국 시간을 반환하는 함수
def get_korea_time():
    """
    현재 한국 시간을 "MM-DD HH:MM:SS" 형식의 문자열로 반환합니다.
    """
    korea_tz = pytz.timezone('Asia/Seoul')
    return datetime.now(korea_tz).strftime("%m-%d %H:%M:%S")

# 벡터 환경 생성 함수
def make_env(env_id, render=False, max_episode_steps=1000):
    """
    환경을 생성하는 함수입니다.
    
    Args:
        env_id: 환경 ID (예: "ALE/Breakout-v5")
        render: 렌더링 여부
        max_episode_steps: 에피소드 최대 길이
        
    Returns:
        환경 생성 함수
    """
    def _thunk():
        env = gym.make(env_id, render_mode="rgb_array" if render else None)
        # 전처리 래퍼 적용
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        # 프레임 스택 래퍼 적용 (4개 프레임 스택)
        env = gym.wrappers.FrameStack(env, 4)
        # 타임아웃 래퍼 추가
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env
    return _thunk

# GAE(Generalized Advantage Estimation) 계산 함수
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """
    GAE(Generalized Advantage Estimation)를 계산합니다.
    
    Args:
        rewards: [step] 크기의 보상 텐서
        values: [step] 크기의 가치 텐서
        dones: [step] 크기의 종료 상태 텐서
        next_value: [env] 크기의 다음 상태 가치 텐서 (마지막 상태)
        gamma: 할인 계수
        lam: GAE 람다 계수
        
    Returns:
        returns: [step] 크기의 반환값 텐서
        advantages: [step] 크기의 이점 텐서
    """
    # 장치 일관성 확인 및 동일한 장치로 이동
    device = rewards.device
    rewards = rewards.to(device)
    values = values.to(device)
    dones = dones.to(device)
    next_value = next_value.to(device)
    
    # 환경 수 확인
    num_envs = next_value.shape[0]
    steps_per_env = len(rewards) // num_envs
    
    # 텐서 재구성
    rewards = rewards.view(steps_per_env, num_envs)
    values = values.view(steps_per_env, num_envs)
    dones = dones.view(steps_per_env, num_envs)
    
    # 가치, 리턴, 이점 초기화
    advantages = torch.zeros_like(values, device=device)
    returns = torch.zeros_like(values, device=device)
    
    # 최종 이점과 가치 설정
    last_advantage = torch.zeros(num_envs, device=device)
    last_value = next_value
    
    # 역순으로 계산
    for t in reversed(range(steps_per_env)):
        # 마스크 계산 (종료 상태가 아닌 경우 1, 종료 상태인 경우 0)
        mask = 1.0 - dones[t]
        
        # 델타 계산: r + gamma * V(s') - V(s)
        delta = rewards[t] + gamma * last_value * mask - values[t]
        
        # 이점 계산: delta + gamma * lambda * mask * A(s')
        advantages[t] = delta + gamma * lam * mask * last_advantage
        
        # 다음 단계를 위한 값 갱신
        last_advantage = advantages[t]
        last_value = values[t]
    
    # 리턴 계산: 이점 + 가치
    returns = advantages + values
    
    # 원래 형태로 다시 펼치기
    advantages = advantages.view(-1)
    returns = returns.view(-1)
    
    return returns, advantages
