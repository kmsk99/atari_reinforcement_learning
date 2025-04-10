# 라이브러리 임포트
import math
import random
import argparse
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 이전 프레임을 저장하기 위한 큐
from collections import deque

from utils import (
    load_checkpoint,
    plot_scores,
    preprocess,
    save_checkpoint,
    PrioritizedReplayBuffer,
    visualize_filters,
    visualize_layer_output,
)

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터 설정
gamma = 0.99  # 할인계수
buffer_limit = 50_000  # 버퍼 크기
batch_size = 32  # 배치 크기

# 현재 스크립트의 경로를 찾습니다.
current_script_path = os.path.dirname(os.path.realpath(__file__))

# 체크포인트 저장 폴더 경로를 현재 스크립트 위치를 기준으로 설정합니다.
checkpoint_dir = os.path.join(current_script_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(os.path.join(current_script_path, "filters"), exist_ok=True)
os.makedirs(os.path.join(current_script_path, "layers"), exist_ok=True)


class Qnet(nn.Module):
    def __init__(self, num_actions):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # 입력 채널이 4 (84x84x4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(self._feature_size(), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _feature_size(self):
        # CNN 출력 크기 계산을 위한 임시 데이터 처리
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
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_action(self, obs, epsilon):
        # 행동 선택 메서드
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()  # NumPy 배열을 Tensor로 변환
        obs = obs.to(device)  # 장치(GPU 또는 CPU)로 이동

        if obs.dim() == 3:  # 단일 프레임인 경우
            obs = obs.unsqueeze(0)  # 배치 차원 추가

        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, out.size(-1) - 1)  # 무작위 액션 선택
        else:
            return out.argmax().item()  # 최적의 액션 선택

    # 중간 레이어 출력 캡처를 위한 훅 등록
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
def train(q, q_target, memory, optimizer):
    for i in range(10):
        # 메모리에서 배치를 샘플링
        s, a, r, s_prime, done_mask, indices = memory.sample(batch_size)

        # GPU로 이동
        s = s.float().to(device)
        s_prime = s_prime.float().to(device)
        r = r.float().to(device)
        done_mask = done_mask.float().to(device)
        a = a.to(device)

        q_out = q(s)
        q_a = q_out.gather(1, a)

        # Double DQN 구현
        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
        max_q_prime = q_target(s_prime).gather(1, argmax_Q)

        target = r + gamma * max_q_prime * done_mask

        # Huber 손실 함수 계산 및 역전파
        loss = F.smooth_l1_loss(q_a, target)  # Huber 손실 사용

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TD 오류 계산
        with torch.no_grad():
            td_error = torch.abs(q_a - target)

        # 우선순위 업데이트
        for idx, error in zip(indices, td_error):
            memory.update_priority(idx, error.item())


# 메인 함수 정의
def main(render):
    env = gym.make("ALE/Breakout-v5", render_mode="human" if render else None)  # 환경 초기화
    # 모델을 GPU로 이동
    q = Qnet(env.action_space.n).to(device)
    q_target = Qnet(env.action_space.n).to(device)
    q_target.load_state_dict(q.state_dict())

    q.register_hooks(["conv1", "conv2", "conv3"])  # 훅 등록

    frame_queue = deque(maxlen=4)

    # 프레임을 초기화하는 함수
    def init_frames():
        for _ in range(4):
            frame_queue.append(torch.zeros(84, 84))

    # memory = ReplayBuffer() 대신에 아래 코드 사용
    memory = PrioritizedReplayBuffer(buffer_limit, device)

    print_interval = 10
    target_update_interval = 10
    train_interval = 4
    score = 0.0
    average_score = 0.0

    # 최적화 함수 설정
    optimizer = optim.RMSprop(q.parameters(), lr=0.001)  # 초기 학습률 설정

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

    if render:
        # 렌더링 모드일 때의 로직
        for _ in range(1):  # 한 번만 실행
            s, _ = env.reset()
            init_frames()
            frame_queue.append(preprocess(s))
            done = False
            plot_scores(scores, "score_plot.png")
            while not done:  # 현재 상태를 4개 프레임으로 구성
                current_state = torch.stack(list(frame_queue), dim=0)

                a = q.sample_action(current_state, 0.0)
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                frame_queue.append(preprocess(s_prime, s))  # 새 프레임 추가
                s = s_prime

                if done:
                    break
    else:
        for n_epi in range(start_episode, 100001):
            epsilon = max(0.01, math.exp(-n_epi / 2000))  # 탐험률 조정
            s, _ = env.reset()
            init_frames()  # 프레임 큐 초기화
            frame_queue.append(preprocess(s))  # 첫 프레임 추가
            done = False

            while not done:
                # 현재 상태를 4개 프레임으로 구성
                current_state = torch.stack(list(frame_queue), dim=0)

                a = q.sample_action(current_state, epsilon)
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                frame_queue.append(preprocess(s_prime, s))  # 새 프레임 추가
                next_state = torch.stack(list(frame_queue), dim=0)

                done_mask = 0.0 if done else 1.0

                memory.put((current_state, a, r, next_state, done_mask))

                s = s_prime

                score += r
                average_score += r
                if done:
                    break

            if memory.size() > 1000:
                if n_epi % train_interval == 0:
                    train(q, q_target, memory, optimizer)

            if n_epi % print_interval == 0:
                print(
                    "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                        n_epi,
                        average_score / print_interval,
                        memory.size(),
                        epsilon * 100,
                    )
                )
                average_score = 0.0

            if n_epi % target_update_interval == 0:
                q_target.load_state_dict(q.state_dict())

            scores.append(score)  # 점수 저장
            score = 0

            # 매 10번째 에피소드마다 그래프 업데이트 및 저장
            if n_epi % 100 == 0:
                plot_scores(scores, "score_plot.png")
                visualize_filters(
                    q, "conv1", epoch=n_epi, save_path=current_script_path
                )
                visualize_filters(
                    q, "conv2", epoch=n_epi, save_path=current_script_path
                )
                visualize_filters(
                    q, "conv3", epoch=n_epi, save_path=current_script_path
                )
                visualize_layer_output(q, "conv1", current_script_path, n_epi)
                visualize_layer_output(q, "conv2", current_script_path, n_epi)
                visualize_layer_output(q, "conv3", current_script_path, n_epi)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment to visualize the agent's performance.",
    )
    args = parser.parse_args()

    main(args.render)
