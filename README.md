# Reinforcement Learning Project - SpaceInvaders

## 소개

이 프로젝트는 강화학습 알고리즘을 이용하여 SpaceInvaders에서 고득점을 얻는 강화학습 모델을 학습시키는 것을 목표로 합니다.

## 시작하기

### 선행 조건

이 프로젝트를 실행하기 위해 다음이 필요합니다:

- Python 3.6 이상
- pytorch-cuda=11.8 이상
- anaconda3 이상

### 설치

1. 이 프로젝트의 저장소를 복제합니다:

   ```
   git clone [저장소 URL]
   ```

2. 필요한 라이브러리를 설치합니다:

   ```
   conda env create -f environment.yml

   혹은

   conda env create -f environment_window.yml
   ```

3. 설치된 라이브러리를 활성화합니다:

   ```
   conda activate [환경 이름]
   ```

### 실행 방법

1. 저장소 디렉토리로 이동합니다:

   ```
   cd [저장소 디렉토리]
   ```

2. 실행할 파일을 파이썬으로 실행합니다:

   ```
   python qnet_v15\atari_qnet_v15.py
   ```

3. 학습된 파일을 테스트하려면 다음과 같이 실행합니다:

   ```
   python qnet_v15\atari_qnet_v15.py --render
   ```
