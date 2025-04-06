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
NUM_ENVS = 8  # 병렬 환경 개수

# 현재 스크립트의 경로를 찾습니다.
current_script_path = os.path.dirname(os.path.realpath(__file__))

# 체크포인트 저장 폴더 경로를 현재 스크립트 위치를 기준으로 설정합니다.
checkpoint_dir = os.path.join(current_script_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(os.path.join(current_script_path, "filters"), exist_ok=True)
os.makedirs(os.path.join(current_script_path, "layers"), exist_ok=True)
os.makedirs(os.path.join(current_script_path, "gameplay"), exist_ok=True)


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
        
        # 추론 시 혼합 정밀도 사용
        with torch.cuda.amp.autocast():
            out = self.forward(obs)
        
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, out.size(-1) - 1)
        else:
            return out.argmax().item()
            
    def sample_actions_batch(self, obs_batch, epsilon):
        """배치 단위로 여러 환경에 대한 행동 샘플링"""
        if not isinstance(obs_batch, torch.Tensor):
            obs_batch = torch.from_numpy(obs_batch).float()
        obs_batch = obs_batch.to(device)
        
        if obs_batch.dim() == 3:
            obs_batch = obs_batch.unsqueeze(0)
        
        # 추론 시 혼합 정밀도 사용
        with torch.cuda.amp.autocast():
            q_values = self.forward(obs_batch)
        
        # 무작위 행동 선택 마스크 생성 (epsilon 확률로 True)
        random_actions = torch.rand(obs_batch.size(0), device=device) < epsilon
        
        # 최적 행동 선택
        greedy_actions = q_values.argmax(dim=1)
        
        # 무작위 행동 생성
        random_action_indices = torch.randint(
            0, q_values.size(-1), (obs_batch.size(0),), device=device
        )
        
        # 최종 행동 선택 (무작위 또는 최적)
        actions = torch.where(random_actions, random_action_indices, greedy_actions)
        
        return actions.cpu().numpy()

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
def train(q, q_target, memory, optimizer, scaler):
    for i in range(10):
        s, a, r, s_prime, done_mask, indices = memory.sample(BATCH_SIZE)

        s = s.float().to(device)
        s_prime = s_prime.float().to(device)
        r = r.float().to(device)
        done_mask = done_mask.float().to(device)
        a = a.to(device)

        # 혼합 정밀도 학습 적용
        with autocast():
            q_out = q(s)
            q_a = q_out.gather(1, a)

            argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
            max_q_prime = q_target(s_prime).gather(1, argmax_Q)

            target = r + GAMMA * max_q_prime * done_mask

            loss = F.smooth_l1_loss(q_a, target)

        # 그래디언트 스케일링 및 역전파
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            td_error = torch.abs(q_a - target)

        for idx, error in zip(indices, td_error):
            memory.update_priority(idx, error.item())


# 벡터 환경 생성 함수
def make_env(env_id, render=False):
    def _thunk():
        env = gym.make(env_id, render_mode="rgb_array" if render else None)
        return env
    return _thunk


# 게임 플레이를 GIF로 저장하는 함수
def save_gameplay_gif(q, episode_num, save_path=os.path.join(current_script_path, "gameplay")):
    """
    현재 학습된 모델을 사용하여 게임 플레이를 GIF로 저장합니다.
    
    Args:
        q: 현재 학습된 Q-네트워크
        episode_num: 현재 에피소드 번호
        save_path: GIF를 저장할 경로
    """
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    frames = []
    s, _ = env.reset()
    
    # 프레임 큐 초기화
    frame_queue = deque(maxlen=1)
    frame_queue.append(preprocess(s))
    
    done = False
    total_reward = 0
    
    # 최대 1000 스텝 또는 게임 종료까지 진행
    for _ in range(1000):
        if done:
            break
            
        current_state = torch.stack(list(frame_queue), dim=0)
        a = q.sample_action(current_state, EXPLORATION_TEST)
        s_prime, r, terminated, truncated, info = env.step(a)
        
        # 렌더링된 프레임 저장
        frames.append(env.render())
        
        done = terminated or truncated
        total_reward += r
        
        # 다음 프레임 저장
        frame_queue.append(preprocess(s_prime, s))
        s = s_prime
    
    env.close()
    
    # GIF 저장
    filename = os.path.join(save_path, f"gameplay_episode_{episode_num}.gif")
    imageio.mimsave(filename, frames, fps=30)
    
    return total_reward


# 메인 함수 정의
def main(render):
    if render:
        # 렌더링 모드에서는 단일 환경만 사용
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        num_envs = 1
    else:
        # 훈련 모드에서는 병렬 환경 사용
        envs = AsyncVectorEnv([make_env("ALE/Breakout-v5") for _ in range(NUM_ENVS)])
        num_envs = NUM_ENVS
    
    # 액션 공간 크기 가져오기
    if render:
        num_actions = env.action_space.n
    else:
        sample_env = gym.make("ALE/Breakout-v5")
        num_actions = sample_env.action_space.n
        sample_env.close()
    
    q = Qnet(num_actions).to(device)
    q_target = Qnet(num_actions).to(device)
    q_target.load_state_dict(q.state_dict())

    q.register_hooks(["conv1", "conv2", "conv3"])

    # 각 환경별로 프레임 큐 초기화
    frame_queues = [deque(maxlen=1) for _ in range(num_envs)]

    # 프레임을 초기화하는 함수
    def init_frames():
        for i in range(num_envs):
            frame_queues[i].append(torch.zeros(84, 84))

    # memory = ReplayBuffer() 대신에 아래 코드 사용
    memory = PrioritizedReplayBuffer(MEMORY_SIZE, device)

    print_interval = 10
    scores = [0.0 for _ in range(num_envs)]  # 각 환경별 점수
    average_score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=0.00025)
    scaler = GradScaler()  # 혼합 정밀도 학습을 위한 GradScaler 초기화

    saved_scores = []
    checkpoint, loaded_scores = load_checkpoint()
    start_episode = 0
    if checkpoint:
        q.load_state_dict(checkpoint["model_state"])
        q_target.load_state_dict(checkpoint["target_model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:  # 이전 체크포인트에 scaler 상태가 없을 수 있으므로 확인
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_episode = checkpoint["episode"]
        saved_scores = loaded_scores

    if render:
        for idx in range(10):
            frames = []
            s, _ = env.reset()
            frame_queues[0].append(preprocess(s))
            done = False
            plot_scores(saved_scores, "score_plot.png")
            while not done:
                current_state = torch.stack(list(frame_queues[0]), dim=0)
                a = q.sample_action(current_state, EXPLORATION_TEST)
                s_prime, r, terminated, truncated, info = env.step(a)
                frames.append(env.render())
                done = terminated or truncated
                frame_queues[0].append(preprocess(s_prime, s))
                s = s_prime

                if done:
                    break

            imageio.mimsave(f"gameplay/Breakout_game_play_{idx+1}.gif", frames, fps=30)

    else:
        # 초기 상태 설정
        states, _ = envs.reset()
        init_frames()
        
        # 각 환경의 현재 프레임을 프레임 큐에 추가
        for i in range(num_envs):
            frame_queues[i].append(preprocess(states[i]))
            
        total_steps = 0
        episode_count = 0
        
        for n_epi in range(start_episode, 100_001):
            epsilon = max(EXPLORATION_MIN, EXPLORATION_MAX * (1 - n_epi / EXPLORATION_STEPS))
            
            # 병렬 환경에서 한 스텝 실행
            # 현재 상태 배치 구성
            current_states = torch.stack([torch.stack(list(frame_queues[i]), dim=0) for i in range(num_envs)])
            
            # 배치 액션 선택
            actions = q.sample_actions_batch(current_states, epsilon)
            
            # 환경 스텝 실행
            next_states, rewards, terminated, truncated, infos = envs.step(actions)
            
            # 완료 여부 확인 (종료 또는 잘림)
            dones = np.logical_or(terminated, truncated)
            
            # 각 환경별로 상태 업데이트 및 메모리에 저장
            for i in range(num_envs):
                # 다음 프레임을 프레임 큐에 추가
                frame_queues[i].append(preprocess(next_states[i], states[i]))
                
                # 현재 상태와 다음 상태 구성
                current_state = torch.stack(list(frame_queues[i]), dim=0)
                next_state = torch.stack(list(frame_queues[i]), dim=0)
                
                # 완료 마스크 계산
                done_mask = 0.0 if dones[i] else 1.0
                
                # 경험을 메모리에 저장
                memory.put((current_state, actions[i], rewards[i], next_state, done_mask), set_priority=(rewards[i] > 0))
                
                # 점수 업데이트
                scores[i] += rewards[i]
                average_score += rewards[i] / num_envs
                
                # 에피소드가 끝났으면 저장하고 재설정
                if dones[i]:
                    saved_scores.append(scores[i])
                    scores[i] = 0.0
                    episode_count += 1
            
            # 현재 상태를 업데이트
            states = next_states
            total_steps += num_envs
            
            # 리플레이 버퍼가 충분히 채워지면 학습 시작
            if memory.size() > REPLAY_START_SIZE and total_steps % TRAINING_FREQUENCY == 0:
                train(q, q_target, memory, optimizer, scaler)
            
            # 주기적으로 진행 상황 출력
            if episode_count > 0 and episode_count % print_interval == 0:
                print(
                    "n_episode: {}, score: {:.1f}, steps: {}, n_buffer: {}, eps: {:.1f}%".format(
                        n_epi,
                        sum(saved_scores[-num_envs:]) / num_envs if saved_scores else 0,
                        total_steps,
                        memory.size(),
                        epsilon * 100,
                    )
                )
                episode_count = 0
                average_score = 0.0
            
            # 타겟 네트워크 업데이트
            if total_steps % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                q_target.load_state_dict(q.state_dict())
            
            # 주기적으로 그래프 및 시각화 업데이트
            if n_epi % 100 == 0:
                plot_scores(saved_scores, "score_plot.png")
                visualize_filters(q, "conv1", epoch=n_epi, save_path=current_script_path)
                visualize_filters(q, "conv2", epoch=n_epi, save_path=current_script_path)
                visualize_filters(q, "conv3", epoch=n_epi, save_path=current_script_path)
                visualize_layer_output(q, "conv1", current_script_path, n_epi)
                visualize_layer_output(q, "conv2", current_script_path, n_epi)
                visualize_layer_output(q, "conv3", current_script_path, n_epi)
            
            # 1000 에피소드마다 현재 학습된 모델의 게임 플레이를 GIF로 저장
            if n_epi % 1000 == 0 and n_epi > 0:
                print(f"에피소드 {n_epi}: 게임 플레이 GIF 저장 중...")
                reward = save_gameplay_gif(q, n_epi)
                print(f"에피소드 {n_epi} 게임 플레이 GIF 저장 완료. 획득 점수: {reward}")
            
            # 주기적으로 모델 저장
            if n_epi % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0 and n_epi != 0:
                save_checkpoint(
                    {
                        "model_state": q.state_dict(),
                        "target_model_state": q_target.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),  # 스케일러 상태 저장
                        "episode": n_epi,
                    },
                    n_epi,
                    saved_scores,
                )

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
