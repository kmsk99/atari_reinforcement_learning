# 라이브러리 임포트
import argparse
import os
import numpy as np
from gymnasium.vector import AsyncVectorEnv
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.distributions import Categorical
from utils import (
    load_checkpoint,
    plot_all_scores,
    save_checkpoint,
    save_gameplay_gif,
    get_korea_time,
    make_env,
    compute_gae,
)

# 현재 스크립트의 경로를 찾습니다
current_script_path = os.path.dirname(os.path.realpath(__file__))

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENT_TAG = "Breakout_PPO_ExpDecayMinLR1e-2_2.5e-4"

# 하이퍼파라미터 설정
# 학습 관련 파라미터
GAMMA = 0.99                # 할인계수: 미래 보상의 현재 가치를 계산할 때 사용, 높을수록 미래 보상 중요시
LAMBDA = 0.95               # GAE 람다: 이점 추정의 편향-분산 트레이드오프 조절, 높을수록 분산 증가
BATCH_SIZE = 32             # 배치 크기: 한 번에 학습하는 샘플 수
NUM_ENVS = 32                # 병렬 환경 개수: 병렬로 실행할 환경 수, 데이터 수집 속도와 다양성 증가
LEARNING_RATE = 0.01     # 학습률: 파라미터 업데이트 크기 결정, 큰 값은 빠른 학습이나 불안정할 수 있음
SCHEDULER_GAMMA = 0.999     # 지수 감쇠 스케줄러의 감마 값: 높을수록 학습률 감소가 느려짐
MIN_LEARNING_RATE = 0.00025    # 최소 학습률: 이 값 이하로 학습률이 감소하지 않음
PPO_EPOCHS = 4              # PPO 업데이트 에포크: 수집된 데이터로 학습할 반복 횟수
PPO_EPSILON = 0.2           # PPO 클리핑 파라미터: 정책 업데이트 제한, 안정성 향상
ENTROPY_COEF = 0.01         # 엔트로피 계수: 탐험 장려 정도, 높을수록 더 많은 탐험
VALUE_COEF = 0.5            # 가치 손실 계수: 가치 함수 학습 중요도
MAX_GRAD_NORM = 0.5         # 그래디언트 클리핑 임계값: 그래디언트 폭발 방지
STEPS_PER_UPDATE = 128     # 업데이트당 스텝 수: 한번의 학습 사이클에 수집할 데이터 양
MAX_EPISODE_STEPS = 27000    # 에피소드 최대 길이: 에피소드 최대 스텝 수 (표준 아타리 환경 설정)

# 환경 및 모델 파라미터
ENV_ID = "ALE/Breakout-v5"  # 환경 ID: 학습할 게임 환경
FRAME_STACK = 4             # 프레임 스택: 입력으로 사용할 연속 프레임 수
FRAME_SIZE = 84             # 프레임 크기: 입력 이미지 크기
CONV1_FILTERS = 32          # Conv1 필터 수
CONV2_FILTERS = 64          # Conv2 필터 수
CONV3_FILTERS = 64          # Conv3 필터 수
FC_SIZE = 512               # 완전 연결 레이어 크기
NUM_UPDATES = 100_001       # 총 업데이트 횟수

# 시각화 및 체크포인트 관련 파라미터
VISUALIZATION_INTERVAL = 500   # 시각화 간격: 몇 에피소드마다 시각화할지
GAMEPLAY_GIF_INTERVAL = 1000    # 게임 플레이 GIF 간격: 몇 에피소드마다 GIF 저장할지
CHECKPOINT_INTERVAL = 1000      # 체크포인트 저장 간격: 몇 에피소드마다 체크포인트 저장할지


class PPONet(nn.Module):
    def __init__(self, num_actions):
        super(PPONet, self).__init__()
        # CNN 특징 추출기
        # 입력 크기: 84x84x4 (4개 프레임 스택)
        # 출력 크기: 32x20x20
        self.conv1 = nn.Conv2d(FRAME_STACK, CONV1_FILTERS, kernel_size=8, stride=4)
        # 입력 크기: 32x20x20
        # 출력 크기: 64x9x9
        self.conv2 = nn.Conv2d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=4, stride=2)
        # 입력 크기: 64x9x9
        # 출력 크기: 64x7x7
        self.conv3 = nn.Conv2d(CONV2_FILTERS, CONV3_FILTERS, kernel_size=3, stride=1)
        
        # 특성 맵 크기 계산
        conv_output_size = CONV3_FILTERS * 7 * 7  # 64 * 7 * 7 = 3136
        
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
            nn.Linear(conv_output_size, FC_SIZE),
            nn.ReLU(),
            nn.Linear(FC_SIZE, num_actions)
        )
        
        # Critic 네트워크 (가치)
        self.critic = nn.Sequential(
            nn.Linear(conv_output_size, FC_SIZE),
            nn.ReLU(),
            nn.Linear(FC_SIZE, 1)
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


# 메인 함수 정의
def main(render, output_dir, checkpoint_dir, filter_dir, layer_dir, gameplay_dir, plot_filename, csv_filename):
    # 시간대 설정 코드 제거 (utils.py의 get_korea_time 사용)
    
    if render:
        env = make_env(ENV_ID, render=True, max_episode_steps=MAX_EPISODE_STEPS, frame_stack=FRAME_STACK, clip_rewards=False)()
        num_envs = 1
    else:        
        # 병렬 환경 생성
        envs = AsyncVectorEnv(
            [make_env(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS, frame_stack=FRAME_STACK, clip_rewards=True) for _ in range(NUM_ENVS)]
        )
        print(f"AsyncVectorEnv 생성 완료: {NUM_ENVS}개 환경")
        num_envs = NUM_ENVS
    
    # 액션 공간 크기 가져오기
    if render:
        num_actions = env.action_space.n
    else:
        sample_env = make_env(ENV_ID)()
        num_actions = sample_env.action_space.n
        sample_env.close()
    
    # PPO 모델 생성
    model = PPONet(num_actions).to(device)
    model.register_hooks(["conv1", "conv2", "conv3"])

    # Optimizer와 Scheduler 초기화
    print(f"초기 학습률: {LEARNING_RATE}, 감마: {SCHEDULER_GAMMA}, 최소 LR: {MIN_LEARNING_RATE}")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)  # Adam epsilon 추가 (안정성)

    # --- 학습률 지수 감쇠 + 최소값 스케줄러 설정 ---
    # 필요한 값들을 람다 함수 스코프 밖에서 정의
    initial_lr = LEARNING_RATE
    gamma_factor = SCHEDULER_GAMMA
    min_lr = MIN_LEARNING_RATE

    # LambdaLR 함수 정의
    # step은 scheduler.step() 호출 횟수 (0부터 시작)
    def lr_lambda_exp_min(step):
        # 지수적으로 감쇠된 목표 학습률 계산
        target_lr = initial_lr * (gamma_factor ** step)
        # 목표 학습률이 최소 학습률보다 작으면 최소 학습률 사용
        actual_lr = max(target_lr, min_lr)
        # 초기 학습률 대비 실제 학습률의 비율(승수) 반환
        return actual_lr / initial_lr

    # LambdaLR 스케줄러 생성
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_exp_min)

    scaler = GradScaler(enabled=torch.cuda.is_available())  # CUDA 사용 가능할 때만 활성화

    # 체크포인트 로드 시도
    saved_scores = []
    checkpoint, loaded_scores = load_checkpoint(checkpoint_dir=checkpoint_dir)
    start_episode = 0
    episode_count = 0
    
    if checkpoint:
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"스케줄러 상태 로드 완료. Last LR: {scheduler.get_last_lr()[0]:.2e}")
        if "episode" in checkpoint:
            start_episode = checkpoint["episode"]
            episode_count = start_episode
        if loaded_scores is not None and len(loaded_scores) > 0:
            saved_scores = loaded_scores
            print(f"이전 점수 기록 로드 완료: {len(saved_scores)}개")
        else:
            print("이전 점수 기록이 없습니다. 새로운 학습을 시작합니다.")
            saved_scores = []
    else:
        print("체크포인트가 없습니다. 새로운 학습을 시작합니다.")
        saved_scores = []

    print(f"체크포인트 로드 완료. 에피소드 {start_episode}부터 시작합니다.")

    if render:
        # 렌더링 모드일 때 평가 수행
        for idx in range(10):
            frames = []
            obs, _ = env.reset()
            
            done = False
            total_reward = 0
            
            plot_all_scores(saved_scores, plot_filename, csv_filename)
            
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
            imageio.mimsave(os.path.join(gameplay_dir, f"Breakout_game_play_{idx+1}.gif"), frames, fps=30)

    else:
        # 학습 모드
        # 초기 상태 설정
        states, _ = envs.reset()
        
        # 훈련 통계 초기화
        total_steps = 0
        episode_rewards = [0.0 for _ in range(num_envs)]  # 각 환경별 보상 합계
        episode_lengths = [0 for _ in range(num_envs)]  # 각 환경별 에피소드 길이
        
        # 메인 학습 루프 (num_updates 사용)
        print(f"총 업데이트 수행 예정: {NUM_UPDATES}")
        start_update_num = start_episode # 체크포인트에서 로드한 업데이트 번호
        if start_update_num > 0:
            print(f"학습 재개: 업데이트 {start_update_num+1}부터 시작.")
        
        for update in range(start_update_num, NUM_UPDATES):
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
                    
                    # 에피소드 최대 길이 초과 시 강제 종료
                    if episode_lengths[i] >= MAX_EPISODE_STEPS:
                        dones_np[i] = True
                        current_time = get_korea_time()
                        print(f"[{current_time}] 환경 {i}에서 최대 길이 도달로 강제 종료")
                    
                    # 에피소드가 끝났으면 통계 저장하고 재설정
                    if dones_np[i]:
                        saved_scores.append(episode_rewards[i])
                        
                        # 에피소드 통계 변수 추가
                        if not hasattr(envs, 'episode_stats'):
                            envs.episode_stats = {
                                'rewards': [],
                                'lengths': [],
                                'last_log_episode': 0
                            }
                        
                        # 현재 에피소드 통계 저장
                        envs.episode_stats['rewards'].append(episode_rewards[i])
                        envs.episode_stats['lengths'].append(episode_lengths[i])
                        
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                        episode_count += 1
                        
                        # 100개 에피소드마다 통합 로그 출력
                        if episode_count % 100 == 0 and episode_count > envs.episode_stats['last_log_episode']:
                            current_time = get_korea_time()
                            # 마지막 100개 에피소드에 대한 통계 계산
                            last_100_rewards = envs.episode_stats['rewards'][-100:]
                            last_100_lengths = envs.episode_stats['lengths'][-100:]
                            
                            avg_reward = np.mean(last_100_rewards)
                            max_reward = np.max(last_100_rewards)
                            min_reward = np.min(last_100_rewards)
                            avg_length = np.mean(last_100_lengths)
                            
                            print(f"[{current_time}] Episodes {episode_count-99}-{episode_count} 요약:")
                            print(f"  평균 점수: {avg_reward:.1f}, 최고 점수: {max_reward:.1f}, 최저 점수: {min_reward:.1f}")
                            print(f"  평균 길이: {avg_length:.1f}, 총 에피소드: {episode_count}")
                            
                            envs.episode_stats['last_log_episode'] = episode_count
                        
                        # 에피소드 수에 기반하여 그래프 그리기와 시각화 체크
                        # 정확한 간격에서만 실행하도록 수정
                        if episode_count % VISUALIZATION_INTERVAL == 0:
                            # 마지막 시각화 이후 충분한 에피소드가 지났는지 확인
                            if episode_count - last_plot_episode >= VISUALIZATION_INTERVAL:
                                last_plot_episode = episode_count
                                print(f"Episode {episode_count}: 그래프 및 시각화 업데이트 중...")
                                plot_all_scores(saved_scores, plot_filename, csv_filename)
                                
                                # 임의의 입력을 사용하여 레이어 활성화 시각화를 위한 순전파 수행
                                # 벡터 환경에서 첫 번째 환경의 현재 상태 사용
                                # sample_state = torch.FloatTensor(np.array(states[0])).unsqueeze(0).to(device)
                                # with torch.no_grad():
                                    # 순전파를 수행하여 활성화 값 생성
                                    # model(sample_state)
                                
                                # 시각화 호출
                                # visualize_filters(model, "conv1", episode_count, save_path=output_dir)
                                # visualize_filters(model, "conv2", episode_count, save_path=output_dir)
                                # visualize_filters(model, "conv3", episode_count, save_path=output_dir)
                                # visualize_layer_output(model, "conv1", output_dir, episode_count)
                                # visualize_layer_output(model, "conv2", output_dir, episode_count)
                                # visualize_layer_output(model, "conv3", output_dir, episode_count)
                                print(f"Episode {episode_count}: 그래프 및 시각화 업데이트 완료")
                        
                        # 에피소드 수에 기반하여 GIF 저장 체크
                        # 정확한 간격에서만 실행하도록 수정
                        if episode_count % GAMEPLAY_GIF_INTERVAL == 0 and episode_count > 0:
                            # 마지막 GIF 저장 이후 충분한 에피소드가 지났는지 확인
                            if episode_count - last_gif_episode >= GAMEPLAY_GIF_INTERVAL:
                                last_gif_episode = episode_count
                                print(f"에피소드 {episode_count}: 게임 플레이 GIF 저장 중...")
                                reward = save_gameplay_gif(model, ENV_ID, episode_count, max_steps=MAX_EPISODE_STEPS, device=device, save_path=gameplay_dir)
                                print(f"에피소드 {episode_count} 게임 플레이 GIF 저장 완료. 획득 점수: {reward}")
                                
                                # 동시에 체크포인트도 저장
                                save_checkpoint(
                                    {
                                        "model_state": model.state_dict(),
                                        "optimizer_state": optimizer.state_dict(),
                                        "scaler_state": scaler.state_dict(),
                                        "scheduler_state": scheduler.state_dict(),
                                        "episode": update,
                                    },
                                    episode_count,
                                    saved_scores,
                                    checkpoint_dir=checkpoint_dir
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
            
            # GAE 계산
            returns, advantages = compute_gae(
                rewards, values, dones, next_value, gamma=GAMMA, lam=LAMBDA
            )
            
            # 이점 정규화
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO 업데이트
            train_ppo(model, optimizer, scaler, observations, actions, log_probs, returns, advantages)
            
            # 학습률 스케줄러 업데이트
            scheduler.step()
            
            # 터미널 출력에 학습 진행 상황 표시
            if update % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                current_time = get_korea_time()
                recent_scores = saved_scores[-10:] if len(saved_scores) > 0 else [0.0]
                avg_score = np.mean(recent_scores)
                print(f"[{current_time}] 업데이트 {update}: 스텝 {total_steps}, 에피소드 {episode_count}, 최근 10개 에피소드 평균 점수: {avg_score:.1f}, 현재 학습률: {current_lr:.6f}")
            
            # 상태 업데이트
            states = next_states

        if not render:
            envs.close()
        else:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render", action="store_true", help="Render the environment."
    )
    args = parser.parse_args()

    # Windows에서 multiprocessing 문제 해결을 위한 설정
    import multiprocessing
    # Windows에서는 'spawn' 방식을 사용해야 함
    multiprocessing.set_start_method('spawn', force=True)
    
    # --- 실험 추적 설정 ---
    # 기본 경로 설정 (ppo_breakout.py 파일이 있는 위치 기준)
    base_results_dir = os.path.join(os.path.dirname(current_script_path), "results")
    
    # 결과 폴더 경로 생성 (타임스탬프 없이 태그만 사용)
    output_dir = os.path.join(base_results_dir, EXPERIMENT_TAG)

    # 결과 폴더 및 하위 폴더 생성
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    filter_dir = os.path.join(output_dir, "filters")
    layer_dir = os.path.join(output_dir, "layers")
    gameplay_dir = os.path.join(output_dir, "gameplay")
    plot_filename = os.path.join(output_dir, "training_")
    csv_filename = os.path.join(output_dir, "scores.csv")

    os.makedirs(base_results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(filter_dir, exist_ok=True)
    os.makedirs(layer_dir, exist_ok=True)
    os.makedirs(gameplay_dir, exist_ok=True)

    print(f"--- 실험 시작: {EXPERIMENT_TAG} ---")
    print(f"결과 저장 경로: {output_dir}")
    
    # 수정된 main 함수 호출 (모든 경로 전달)
    main(args.render, output_dir, checkpoint_dir, filter_dir, layer_dir, gameplay_dir, plot_filename, csv_filename)
