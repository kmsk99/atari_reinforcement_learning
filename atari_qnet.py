# libraries
import gymnasium as gym
import collections
import random
import numpy as np

# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000  # size of replay buffer
batch_size = 32


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  # double-ended queue

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s.transpose(2, 0, 1))
            s_prime_lst.append(s_prime.transpose(2, 0, 1))
            a_lst.append([a])
            r_lst.append([r])  # 보상을 리스트에 추가
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(np.array(s_lst), dtype=torch.float),
            torch.tensor(np.array(a_lst), dtype=torch.long),
            torch.tensor(np.array(r_lst), dtype=torch.float),  # 보상 텐서의 데이터 타입을 Float으로 지정
            torch.tensor(np.array(s_prime_lst), dtype=torch.float),
            torch.tensor(np.array(done_mask_lst)),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        # CNN 레이어
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 완전 연결 레이어
        self.fc1 = nn.Linear(self._feature_size(), 512)
        self.fc2 = nn.Linear(512, 6)  # 6개 액션

    def _feature_size(self):
        # 임시 데이터를 통해 CNN 출력 크기 계산
        with torch.no_grad():
            return nn.Sequential(self.conv1, self.conv2, self.conv3).forward(torch.zeros(1, 3, 210, 160)).view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # 평탄화: 'view' 대신 'reshape' 사용
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_action(self, obs, epsilon):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()  # NumPy 배열을 Tensor로 변환
        obs = obs.unsqueeze(0)  # 단일 관찰에 배치 차원 추가
        obs = obs.permute(0, 3, 1, 2)  # 차원 재정렬
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 5)  # 무작위 액션
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # Ensure tensors are of type Float
        s = s.float()
        s_prime = s_prime.float()
        r = r.float()
        done_mask = done_mask.float()

        q_out = q(s)
        q_a = q_out.gather(1, a)

        # Double DQN
        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
        max_q_prime = q_target(s_prime).gather(1, argmax_Q)

        target = r + gamma * max_q_prime * done_mask

        # MSE Loss
        loss = F.mse_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def main():
    env = gym.make("ALE/SpaceInvaders-v5")
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 5
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(1000):
        epsilon = max(
            0.01, 0.08 - 0.01 * (n_epi / 200)
        )  # Linear annealing from 8% to 1%
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
            if done:
                break

        if memory.size() > 1000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            score = 0.0

    env.close()


if __name__ == "__main__":
    main()
