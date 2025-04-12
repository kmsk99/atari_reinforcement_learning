# 라이브러리 임포트
import glob
import re
import torchvision
import os
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd
from datetime import datetime
import pytz
from gymnasium.wrappers import AtariPreprocessing, TransformReward, FrameStack, TimeLimit

# 현재 스크립트의 경로를 찾습니다.
current_script_path = os.path.dirname(os.path.realpath(__file__))

# 체크포인트 저장 폴더 경로를 현재 스크립트 위치를 기준으로 설정합니다.
default_checkpoint_dir = os.path.join(current_script_path, "checkpoints")


def save_checkpoint(
    state, episode, scores, checkpoint_dir=None, keep_last=3
):
    # checkpoint_dir 인자가 없으면 기본값 사용
    if checkpoint_dir is None:
        checkpoint_dir = default_checkpoint_dir
        
    # 체크포인트 디렉토리가 없으면 생성
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
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
def load_checkpoint(checkpoint_dir=None):
    # checkpoint_dir 인자가 없으면 기본값 사용
    if checkpoint_dir is None:
        checkpoint_dir = default_checkpoint_dir
        
    # 체크포인트 디렉토리가 없으면 None 반환
    if not os.path.exists(checkpoint_dir):
        print(f"체크포인트 디렉토리가 없습니다: {checkpoint_dir}")
        return None, None
        
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
def plot_scores(scores, filename, save_csv=True, csv_filename=None, max_points=1000):
    # CSV 파일 저장
    if save_csv and csv_filename is not None:
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
    if csv_filename is not None and os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        scores = df['score'].values
    
    plt.figure(figsize=(12, 6))
    
    # 데이터 포인트 샘플링 (데이터가 너무 많을 경우)
    raw_data_len = len(scores)
    if raw_data_len > max_points:
        step = max(1, raw_data_len // max_points)
        x_indices = np.arange(0, raw_data_len, step)
    else:
        x_indices = np.arange(raw_data_len)
    
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
    
    # 최고 점수 추적 및 플롯 (샘플링된 데이터 사용)
    max_scores = [max(scores[:i+1]) for i in range(len(scores))]
    plt.plot(x_indices, [max_scores[i] for i in x_indices], label="Best Score", color="green", linestyle="--", marker=".", markersize=3)
    
    # 100개 윈도우의 Moving Best 계산 (샘플링된 데이터 사용)
    window_size = 100
    moving_best = []
    for i in range(len(scores)):
        window_start = max(0, i - window_size + 1)
        moving_best.append(max(scores[window_start:i+1]))
    plt.plot(x_indices, [moving_best[i] for i in x_indices], label="Moving Best (window=100)", color="purple", linestyle="-.", marker=".", markersize=3)
    
    # 100개 윈도우의 Moving Worst 계산 (샘플링된 데이터 사용)
    moving_worst = []
    for i in range(len(scores)):
        window_start = max(0, i - window_size + 1)
        moving_worst.append(min(scores[window_start:i+1]))
    plt.plot(x_indices, [moving_worst[i] for i in x_indices], label="Moving Worst (window=100)", color="orange", linestyle="-.", marker=".", markersize=3)

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Scores per Episode")
    plt.legend()

    plt.savefig(filename)
    plt.close()

# 이동 최고/최저가 없는 단순 플롯 함수
def plot_scores_simple(scores, filename, save_csv=False, csv_filename=None, max_points=1000):
    # CSV 파일에서 데이터 읽기
    if csv_filename is not None and os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        scores = df['score'].values
    
    plt.figure(figsize=(12, 6))
    
    # 데이터 포인트 샘플링 (데이터가 너무 많을 경우)
    raw_data_len = len(scores)
    if raw_data_len > max_points:
        step = max(1, raw_data_len // max_points)
        x_indices = np.arange(0, raw_data_len, step)
    else:
        x_indices = np.arange(raw_data_len)
    
    # 실제 점수 (샘플링된 데이터)
    plt.scatter(x_indices, [scores[i] for i in x_indices], color="gray", alpha=0.3, s=10, label="Actual Scores")
    
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
    plt.title("Scores per Episode (Simple)")
    plt.legend()

    plt.savefig(filename)
    plt.close()

# 최근 1000개 에피소드만 표시하는 플롯 함수
def plot_recent_scores(scores, filename, save_csv=False, csv_filename=None, max_points=500):
    # CSV 파일에서 데이터 읽기
    if csv_filename is not None and os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        scores = df['score'].values
    
    # 최근 1000개 에피소드만 선택
    recent_count = min(1000, len(scores))
    recent_scores = scores[-recent_count:]
    
    plt.figure(figsize=(12, 6))
    
    # x축 값 계산 (전체 에피소드 수 기준으로)
    start_episode = len(scores) - recent_count + 1
    
    # 데이터 포인트 샘플링 (데이터가 너무 많을 경우)
    raw_data_len = len(recent_scores)
    if raw_data_len > max_points:
        step = max(1, raw_data_len // max_points)
        indices = np.arange(0, raw_data_len, step)
        x_values = [start_episode + i for i in indices]
        plt.scatter(x_values, [recent_scores[i] for i in indices], color="gray", alpha=0.3, s=10, label="Actual Scores")
    else:
        x_values = list(range(start_episode, len(scores) + 1))
        plt.scatter(x_values, recent_scores, color="gray", alpha=0.3, s=10, label="Actual Scores")
    
    # 10개 이동평균 계산
    moving_avg_10 = [np.mean(recent_scores[max(0, i - 9) : i + 1]) for i in range(len(recent_scores))]
    plt.plot(range(start_episode, len(scores) + 1), moving_avg_10, label="10-episode Moving Avg", color="blue")

    # 100개 이동평균 계산
    moving_avg_100 = [
        np.mean(recent_scores[max(0, i - 99) : i + 1]) for i in range(len(recent_scores))
    ]
    plt.plot(range(start_episode, len(scores) + 1), moving_avg_100, label="100-episode Moving Avg", color="yellow")

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(f"Recent {recent_count} Episodes")
    plt.legend()

    plt.savefig(filename)
    plt.close()

# 세 개의 그래프를 한번에 생성하는 함수
def plot_all_scores(scores, plot_prefix="training_", csv_filename=None, max_points=1000):
    # 모든 그래프 생성
    plot_scores(scores, f"{plot_prefix}scores_full.png", save_csv=True, csv_filename=csv_filename, max_points=max_points)
    plot_scores_simple(scores, f"{plot_prefix}scores_simple.png", save_csv=False, csv_filename=csv_filename, max_points=max_points)
    plot_recent_scores(scores, f"{plot_prefix}scores_recent.png", save_csv=False, csv_filename=csv_filename, max_points=max_points//2)
    
    print(f"세 개의 그래프가 생성되었습니다: {plot_prefix}scores_full.png, {plot_prefix}scores_simple.png, {plot_prefix}scores_recent.png")

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
    
    # 환경 생성 - make_env 함수 활용
    env = make_env(env_id, render=True, max_episode_steps=max_steps, frame_stack=4, clip_rewards=False)()
    
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
    os.makedirs(save_path, exist_ok=True)
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
def make_env(env_id, render=False, max_episode_steps=27000, frame_stack=4, clip_rewards=True):
    """
    환경을 생성하는 함수 (AtariPreprocessing 사용 최적화)

    Args:
        env_id (str): 환경 ID
        render (bool): 렌더링 모드 여부
        max_episode_steps (int): 최대 스텝 수 (TimeLimit)
        frame_stack (int): 프레임 스택 수
        clip_rewards (bool): 보상을 -1, 0, 1로 클리핑할지 여부

    Returns:
        Callable: 환경 생성 함수
    """
    render_mode = "rgb_array" if render else None # GIF 저장을 위해 render=True시 rgb_array 사용
    
    def _thunk():
        try:
            # 환경 생성 - frameskip=1로 설정하여 원본 환경의 프레임 스킵을 비활성화
            # AtariPreprocessing 래퍼가 자체적으로 프레임 스킵을 처리하기 때문
            env = gym.make(env_id, render_mode=render_mode, frameskip=1)
            
            # Atari 전처리 래퍼 적용
            # - noop_max=30: 시작 시 랜덤 No-Op (초기 상태 다양화)
            # - frame_skip=4: 프레임 스킵 (연산 효율성, 일반적 설정)
            # - screen_size=84: 관측 크기 조정
            # - terminal_on_life_loss=True: 목숨 잃으면 done=True 처리 (매우 중요)
            # - grayscale_obs=True: 흑백 변환
            # - scale_obs=True: 관측값을 [0, 1] 범위 float로 자동 스케일링
            env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=True, grayscale_obs=True, scale_obs=True)
            
            # 프레임 스택 래퍼 적용
            env = FrameStack(env, frame_stack)
            
            # 보상 클리핑 래퍼 (-1, 0, 1) 적용 (학습 안정화)
            # if clip_rewards:
            #     env = TransformReward(env, lambda r: np.sign(r))
            
            # 타임 리밋 래퍼 (에피소드 최대 길이 제한)
            if max_episode_steps is not None:
                env = TimeLimit(env, max_episode_steps=max_episode_steps)
                
            return env
            
        except Exception as e:
            print(f"환경 생성 중 오류 발생: {e}")
            raise e
            
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

def save_pretrained_model(model_state, pretrained_dir=None):
    """
    모델의 가중치만 저장하는 pretrained 모델 체크포인트를 생성합니다.
    
    Args:
        model_state: 모델의 state_dict
        pretrained_dir: pretrained 모델을 저장할 디렉토리 (기본값: current_script_path/pretrained)
    """
    # pretrained_dir 인자가 없으면 기본값 사용
    if pretrained_dir is None:
        pretrained_dir = os.path.join(current_script_path, "pretrained")
        
    # pretrained 디렉토리가 없으면 생성
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir, exist_ok=True)
        
    filepath = os.path.join(pretrained_dir, f"pretrained_checkpoint_0.pth")
    torch.save({"model_state": model_state}, filepath)
    print(f"Pretrained 모델이 {filepath}에 저장되었습니다.")
    
    return filepath

def load_pretrained_model(model, pretrained_dir=None, filename="pretrained_checkpoint_0.pth"):
    """
    Pretrained 모델을 불러옵니다.
    
    Args:
        model: 불러올 모델 객체
        pretrained_dir: pretrained 모델이 있는 디렉토리 (기본값: current_script_path/pretrained)
        filename: 불러올 파일 이름
    
    Returns:
        성공적으로 불러왔는지 여부 (bool)
    """
    # pretrained_dir 인자가 없으면 기본값 사용
    if pretrained_dir is None:
        pretrained_dir = os.path.join(current_script_path, "pretrained")
    
    filepath = os.path.join(pretrained_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Pretrained 모델 파일을 찾을 수 없습니다: {filepath}")
        return False
    
    try:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Pretrained 모델을 성공적으로 불러왔습니다: {filepath}")
        return True
    except Exception as e:
        print(f"Pretrained 모델을 불러오는 중 오류 발생: {e}")
        return False
