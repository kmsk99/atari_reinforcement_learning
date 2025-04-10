# 라이브러리 임포트
import math
import random
import argparse
import os
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.cuda.amp import autocast, GradScaler  # 혼합 정밀도 학습을 위한 임포트
from torch.distributions import Categorical
from utils import (
    load_checkpoint,
    plot_scores,
    preprocess,
    save_checkpoint,
    visualize_filters,
    visualize_layer_output,
)

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터 설정
GAMMA = 0.99  # 할인계수
LAMBDA = 0.95  # GAE 람다
BATCH_SIZE = 32  # 배치 크기
NUM_ENVS = 32  # 병렬 환경 개수
LEARNING_RATE = 0.00025  # 학습률
PPO_EPOCHS = 4  # PPO 업데이트 에포크
PPO_EPSILON = 0.2  # PPO 클리핑 파라미터
ENTROPY_COEF = 0.01  # 엔트로피 계수
VALUE_COEF = 0.5  # 가치 손실 계수
MAX_GRAD_NORM = 0.5  # 그래디언트 클리핑 임계값
STEPS_PER_UPDATE = 128  # 업데이트당 스텝 수

# 현재 스크립트의 경로를 찾습니다.
current_script_path = os.path.dirname(os.path.realpath(__file__))

# 체크포인트 저장 폴더 경로를 현재 스크립트 위치를 기준으로 설정합니다.
checkpoint_dir = os.path.join(current_script_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(os.path.join(current_script_path, "filters"), exist_ok=True)
os.makedirs(os.path.join(current_script_path, "layers"), exist_ok=True)
os.makedirs(os.path.join(current_script_path, "gameplay"), exist_ok=True)


class PPONet(nn.Module):
    def __init__(self, num_actions):
        super(PPONet, self).__init__()
        # CNN 특징 추출기 (기존과 동일)
        # 입력 크기: 84x84x4 (4개 프레임 스택)
        # 출력 크기: 32x20x20
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # 입력 크기: 32x20x20
        # 출력 크기: 64x9x9
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 입력 크기: 64x9x9
        # 출력 크기: 64x7x7
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 출력 특성 크기: 64 * 7 * 7 = 3136
        self.flatten = nn.Flatten()
        
        # 특징 추출기
        self.feature_extractor = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            self.flatten
        )
        
        # Actor 네트워크 (정책)
        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Critic 네트워크 (가치)
        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.activations = {}  # 시각화를 위한 활성화 저장

    def forward(self, x):
        features = self.feature_extractor(x)
        action_logits = self.actor(features)
        state_values = self.critic(features)
        return action_logits, state_values
    
    def get_action_and_value(self, obs, action=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(device)
        
        # 추론 시 혼합 정밀도 사용
        with torch.cuda.amp.autocast():
            logits, values = self.forward(obs)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        # values가 [batch, 1] 형태인지 확인하고 필요시 조정
        if values.dim() > 1 and values.size(-1) == 1:
            values = values.squeeze(-1)
        
        return action, log_prob, entropy, values

    def register_hooks(self, layer_names):
        self.activations = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        for name in layer_names:
            layer = getattr(self, name)
            layer.register_forward_hook(get_activation(name))


# PPO 업데이트 함수
def train_ppo(model, optimizer, scaler, observations, actions, log_probs, returns, advantages):
    # 데이터셋을 배치로 나누어 여러 번 훈련
    for _ in range(PPO_EPOCHS):
        # 데이터 전체를 미니배치로 나누기
        mini_batch_size = observations.size(0) // (BATCH_SIZE // NUM_ENVS)
        
        # 미니배치 크기가 0인 경우 방지
        if mini_batch_size <= 0:
            mini_batch_size = observations.size(0)
            
        indices = torch.randperm(observations.size(0))
        
        # 미니배치 단위로 처리
        for start in range(0, observations.size(0), mini_batch_size):
            # 배치 인덱스 추출
            end = min(start + mini_batch_size, observations.size(0))
            mini_batch_indices = indices[start:end]
            
            # 미니배치 데이터 추출
            mb_obs = observations[mini_batch_indices].to(device)
            mb_actions = actions[mini_batch_indices].to(device)
            mb_old_log_probs = log_probs[mini_batch_indices].to(device)
            mb_returns = returns[mini_batch_indices].to(device)
            mb_advantages = advantages[mini_batch_indices].to(device)
            
            # 혼합 정밀도 학습 적용
            with autocast():
                # 현재 정책에서의 행동 확률 및 가치 계산
                _, new_log_probs, entropy, values = model.get_action_and_value(mb_obs, mb_actions)
                
                # PPO 비율 계산
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # PPO 손실 함수 계산 (클리핑)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic 손실 계산
                critic_loss = F.mse_loss(values, mb_returns)
                
                # 엔트로피 보너스
                entropy_loss = -entropy.mean()
                
                # 전체 손실 계산
                loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss
            
            # 그래디언트 스케일링 및 역전파
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # 그래디언트 클리핑
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            scaler.step(optimizer)
            scaler.update()


# 벡터 환경 생성 함수
def make_env(env_id, render=False):
    def _thunk():
        env = gym.make(env_id, render_mode="rgb_array" if render else None)
        # 전처리 래퍼 적용
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        # 프레임 스택 래퍼 적용 (4개 프레임 스택)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return _thunk


# 게임 플레이를 GIF로 저장하는 함수
def save_gameplay_gif(model, episode_num, save_path=os.path.join(current_script_path, "gameplay")):
    """
    현재 학습된 모델을 사용하여 게임 플레이를 GIF로 저장합니다.
    
    Args:
        model: 현재 학습된 PPO 모델
        episode_num: 현재 에피소드 번호
        save_path: GIF를 저장할 경로
    """
    # 전처리와 프레임 스택이 적용된 환경 생성
    env = make_env("ALE/Breakout-v5", render=True)()
    frames = []
    obs, _ = env.reset()
    
    done = False
    total_reward = 0
    
    # 최대 1000 스텝 또는 게임 종료까지 진행
    for _ in range(1000):
        if done:
            break
            
        # obs는 이미 스택된 프레임 형태이므로 바로 사용
        # numpy 배열을 PyTorch 텐서로 변환
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
    filename = os.path.join(save_path, f"gameplay_episode_{episode_num}.gif")
    imageio.mimsave(filename, frames, fps=30)
    
    return total_reward


# 일반화된 이득 추정(GAE) 계산 함수
def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=LAMBDA):
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
    
    # 텐서 형태 출력 (디버깅용)
    # print(f"rewards shape: {rewards.shape}, values shape: {values.shape}")
    # print(f"dones shape: {dones.shape}, next_value shape: {next_value.shape}")
    
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


# 메인 함수 정의
def main(render):
    if render:
        # 렌더링 모드에서는 단일 환경만 사용 (프레임 스택 적용)
        env = make_env("ALE/Breakout-v5", render=True)()
        num_envs = 1
    else:
        # 훈련 모드에서는 병렬 환경 사용 (각 환경에 프레임 스택 적용)
        envs = AsyncVectorEnv([make_env("ALE/Breakout-v5") for _ in range(NUM_ENVS)])
        num_envs = NUM_ENVS
    
    # 액션 공간 크기 가져오기
    if render:
        num_actions = env.action_space.n
    else:
        sample_env = make_env("ALE/Breakout-v5")()
        num_actions = sample_env.action_space.n
        sample_env.close()
    
    # PPO 모델 생성
    model = PPONet(num_actions).to(device)
    model.register_hooks(["conv1", "conv2", "conv3"])

    # Optimizer와 Scheduler 초기화
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()  # 혼합 정밀도 학습을 위한 GradScaler 초기화

    # 체크포인트 로드 시도
    saved_scores = []
    checkpoint, loaded_scores = load_checkpoint()
    start_episode = 0
    episode_count = 0
    
    if checkpoint:
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if "episode" in checkpoint:
            start_episode = checkpoint["episode"]
            episode_count = start_episode  # 에피소드 카운터도 체크포인트 에피소드 번호로 초기화
        if loaded_scores is not None:
            saved_scores = loaded_scores
            
    print(f"체크포인트 로드 완료. 에피소드 {start_episode}부터 시작합니다.")

    if render:
        # 렌더링 모드일 때 평가 수행
        for idx in range(10):
            frames = []
            obs, _ = env.reset()
            
            done = False
            total_reward = 0
            
            plot_scores(saved_scores, "score_plot.png")
            
            while not done:
                # obs는 이미 스택된 프레임 형태이므로 바로 사용
                obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)
                
                # 행동 선택
                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(obs_tensor)
                    action = action.cpu().item()
                
                next_obs, r, terminated, truncated, info = env.step(action)
                frames.append(env.render())
                
                done = terminated or truncated
                total_reward += r
                
                obs = next_obs

            print(f"평가 에피소드 {idx+1} 완료. 획득 점수: {total_reward}")
            imageio.mimsave(f"gameplay/Breakout_game_play_{idx+1}.gif", frames, fps=30)

    else:
        # 학습 모드
        # 초기 상태 설정
        states, _ = envs.reset()
        
        # 훈련 통계 초기화
        total_steps = 0
        episode_rewards = [0.0 for _ in range(num_envs)]  # 각 환경별 보상 합계
        episode_lengths = [0 for _ in range(num_envs)]  # 각 환경별 에피소드 길이
        
        # 메인 학습 루프
        for n_epi in range(start_episode, 100_001):
            # 데이터 수집을 위한 임시 버퍼
            observations = []
            actions = []
            log_probs = []
            values = []
            rewards = []
            dones = []
            
            # 그래프 그리기와 GIF 저장을 위한 마지막 확인 에피소드 추적
            last_plot_episode = 0
            last_gif_episode = 0
            
            # 시각화 간격 설정 (에피소드 단위)
            VISUALIZATION_INTERVAL = 100
            GAMEPLAY_GIF_INTERVAL = 500

            # STEPS_PER_UPDATE 스텝 동안 데이터 수집
            for step in range(STEPS_PER_UPDATE):
                total_steps += num_envs
                
                # 상태를 PyTorch 텐서로 변환
                # states는 벡터 환경의 출력으로, 각 환경마다 이미 4개의 프레임이 스택된 상태
                current_states = torch.FloatTensor(np.array(states)).to(device)
                
                # 모델로부터 행동, 로그 확률, 엔트로피, 가치 얻기
                with torch.no_grad():
                    actions_tensor, log_probs_tensor, entropy, values_tensor = model.get_action_and_value(current_states)
                
                # CPU로 이동 후 numpy 변환 (환경 스텝용)
                actions_np = actions_tensor.cpu().numpy()
                
                # 환경에서 한 스텝 실행
                next_states, rewards_np, terminated, truncated, infos = envs.step(actions_np)
                
                # 완료 여부 확인 (종료 또는 잘림)
                dones_np = np.logical_or(terminated, truncated)
                
                # 에피소드 통계 업데이트
                for i in range(num_envs):
                    episode_rewards[i] += rewards_np[i]
                    episode_lengths[i] += 1
                    
                    # 에피소드가 끝났으면 통계 저장하고 재설정
                    if dones_np[i]:
                        saved_scores.append(episode_rewards[i])
                        print(f"Episode {episode_count}: 환경 {i}에서 점수 {episode_rewards[i]:.1f}, 길이 {episode_lengths[i]}")
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                        episode_count += 1
                        
                        # 에피소드 수에 기반하여 그래프 그리기와 시각화 체크
                        # 정확한 간격에서만 실행하도록 수정
                        if episode_count % VISUALIZATION_INTERVAL == 0:
                            # 마지막 시각화 이후 충분한 에피소드가 지났는지 확인
                            if episode_count - last_plot_episode >= VISUALIZATION_INTERVAL:
                                last_plot_episode = episode_count
                                print(f"Episode {episode_count}: 그래프 및 시각화 업데이트 중...")
                                plot_scores(saved_scores, "score_plot.png")
                                
                                # 임의의 입력을 사용하여 레이어 활성화 시각화를 위한 순전파 수행
                                # 벡터 환경에서 첫 번째 환경의 현재 상태 사용
                                sample_state = torch.FloatTensor(np.array(states[0])).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    # 순전파를 수행하여 활성화 값 생성
                                    model(sample_state)
                                
                                # 시각화 호출
                                visualize_filters(model, "conv1", epoch=episode_count, save_path=current_script_path)
                                visualize_filters(model, "conv2", epoch=episode_count, save_path=current_script_path)
                                visualize_filters(model, "conv3", epoch=episode_count, save_path=current_script_path)
                                visualize_layer_output(model, "conv1", current_script_path, episode_count)
                                visualize_layer_output(model, "conv2", current_script_path, episode_count)
                                visualize_layer_output(model, "conv3", current_script_path, episode_count)
                                print(f"Episode {episode_count}: 그래프 및 시각화 업데이트 완료")
                        
                        # 에피소드 수에 기반하여 GIF 저장 체크
                        # 정확한 간격에서만 실행하도록 수정
                        if episode_count % GAMEPLAY_GIF_INTERVAL == 0 and episode_count > 0:
                            # 마지막 GIF 저장 이후 충분한 에피소드가 지났는지 확인
                            if episode_count - last_gif_episode >= GAMEPLAY_GIF_INTERVAL:
                                last_gif_episode = episode_count
                                print(f"에피소드 {episode_count}: 게임 플레이 GIF 저장 중...")
                                reward = save_gameplay_gif(model, episode_count)
                                print(f"에피소드 {episode_count} 게임 플레이 GIF 저장 완료. 획득 점수: {reward}")
                                
                                # 동시에 체크포인트도 저장
                                save_checkpoint(
                                    {
                                        "model_state": model.state_dict(),
                                        "optimizer_state": optimizer.state_dict(),
                                        "scaler_state": scaler.state_dict(),
                                        "episode": n_epi,
                                    },
                                    episode_count,
                                    saved_scores,
                                )
                                print(f"에피소드 {episode_count}: 체크포인트 저장 완료")
                
                # 데이터 저장 - 모두 같은 장치(CPU)에 저장
                observations.append(current_states.cpu())
                actions.append(actions_tensor.cpu())
                log_probs.append(log_probs_tensor.cpu())
                values.append(values_tensor.cpu())
                rewards.append(torch.FloatTensor(rewards_np).cpu())
                dones.append(torch.FloatTensor(dones_np).cpu())
                
                # 상태 업데이트
                states = next_states
            
            # 마지막 상태의 가치 계산 (부트스트래핑)
            final_states = torch.FloatTensor(np.array(states)).to(device)
            with torch.no_grad():
                _, _, _, next_value = model.get_action_and_value(final_states)
                next_value = next_value.cpu()  # CPU로 이동
            
            # 버퍼의 모든 데이터를 텐서로 변환 (이미 모두 CPU에 있음)
            observations = torch.cat(observations)
            actions = torch.cat(actions)
            log_probs = torch.cat(log_probs)
            values = torch.cat(values)
            rewards = torch.cat(rewards)
            dones = torch.cat(dones)
            
            # GAE 계산 (모든 텐서가 CPU에 있는 상태)
            returns, advantages = compute_gae(
                rewards, values, dones, next_value
            )
            
            # 이점 정규화
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO 업데이트
            train_ppo(model, optimizer, scaler, observations, actions, log_probs, returns, advantages)
            
            # 터미널 출력에 학습 진행 상황 표시
            if n_epi % 10 == 0:
                print(f"업데이트 {n_epi}: 스텝 {total_steps}, 에피소드 {episode_count}, 최근 10개 에피소드 평균 점수: {np.mean(saved_scores[-10:]):.1f}")
            
            # 상태 업데이트
            states = next_states

        if not render:
            envs.close()
        else:
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
