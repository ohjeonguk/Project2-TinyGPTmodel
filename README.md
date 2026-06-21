# Project2-GPT2.0 model
 이 프로젝트는 gpt의 핵심 아이디어를 간단한 형태로 직접 구현한 모델이다. 모델은 input.txt에 들어 있는 글을 학습한 뒤, 앞에 나온 문자들을 바탕으로 다음 문자를 예측한다.

# 1. 프로젝트 목표
이 프로젝트의 목표는 다음과 같다.
1. 텍스트 데이터를 문자 단위로 토큰화한다.
2. 각 문자를 정수 id로 변환한다.
3. 입력 sequence를 이용하여 다음 문자를 예측하는 모델을 만든다.
4. Transfomer 기반의 작은 gpt 모델을 구현한다.
5. 학습된 모델을 이용하여 새로운 글을 생성한다.

# 2. 프로젝트 흐름
이 프로젝트의 전체 흐름은 다음과 같다.

input.txt 읽기
→ 문자 단위 vocabulary 생성
→ 문자를 정수 id로 변환
→ 학습 데이터와 검증 데이터 분리
→ 입력 x와 정답 y batch 생성
→ TinyGPT 모델 학습
→ 다음 문자 예측
→ 새로운 글 생성

# 3. 문자 단위 토큰화
프로젝트에서는 단어가 아니라 문자 하나하나를 토큰으로 사용한다.
각 문자는 vocabulary에 등록되고, 고유한 정수 id를 가진다.
모델은 문자를 직접 이해하는 것이 아니라, 이 정수 id들을 입력으로 받아 학습한다.

# 4. 모델 구조
구현한 gpt2.0 모델은 다음 요소로 구성된다.

4.1 Token Embedding

문자 id는 그 자체로는 의미 있는 벡터가 아니기 때문에, 먼저 embedding layer를 통해 고차원 벡터로 변환한다.
    문자 id -> token embedding vector
이를 통해 모델은 각 문자를 학습 가능한 벡터 표현으로 다룰 수 있다.

4.2 Position Embedding

Transformer는 RNN과 달리 입력을 순서대로 처리하지 않는다. 따라서 각 문자가 문장 안에서 몇 번째 위치에 있는지 알려주는 정보가 필요하다.
이를 위해 position embedding을 사용한다.
    최종 입력 벡터 = token embedding + position embedding
이 과정을 통해 모델은 문자 자체의 의미뿐만 아니라 문자의 위치 정보도 함께 학습할 수 있다.

4.3 Masked Self-Attention

Self-Attention은 각 문자가 문장 안의 다른 문자들을 얼마나 참고할지 계산하는 구조이다.
하지만 gpt는 다음 문자를 예측하는 모델이므로, 현재 위치에서 미래의 문자를 보면 안 된다.
예를 들어, '나는 밥을 먹었다'를 학습한다고 할 때, '먹'을 예측하는 시점에서 모델이 이미 뒤에 있는 '었', '다'를 보면 정답을 미리 보는 것이므로 미래 위치를 가리는 mask가 필요하다.
이 프로젝트에서는 lower triangular matrix를 이용해 미래 토큰을 보지 못하게 막았다.

4.4 Multi-Head Attention

하나의 attention head만 사용하면 문맥을 한 가지 관점에섬나 바라보게 된다.
Multi-head attention은 여러 개의 attention head를 병렬로 사용하여 다양한 관점에서 문맥 정보를 학습한다.
예를 들어 어떤 head는 가까운 문자를 중요하게 볼 수 있고, 다른 head는 문장 전체의 구조를 중요하게 볼 수 있다.
이 프로젝트에서는 여러 attention head의 출력을 이어붙인 뒤, 다시 linear layer를 통과시켜 최종 attention 결과를 만든다.

4.5 FeedForward Network

Attention이 문자들 사이의 관계를 학습하는 부분이라면, feedforward network는 각 위치의 벡터 표현을 더 깊게 변환하는 부분이다.

Linear -> ReLU -> Linear -> Dropout 구조를 통해 모델은 attention으로 얻은 문맥 정보를 더 복잡한 표현으로 바꿀 수 있다.

4.6 Residual Connection

TransForme Block에서는 입력 X에 attention 결과나 feedforward 결과를 바로 더해준다.

    x = x + attention(x)
    x = x + feedforward(x)
    
이 구조를 residual connection이라고 한다.

4.7 Layer Normalization

Layer normailization은 각 층의 입력 분포를 안정화하는 역할을 한다.
이 프로젝트에서는 attention과 feedforward network를 적용하기 전에 layer normalization을 사용하였다.

    x = x + attention(layer_norm(x))
    x = x + feedforward(layer_norm(x))

이 구조는 학습을 더 안정적으로 만들어준다.

# 5. 학습 방식
모델은 다음 문자 예측 문자를 풀도록 학습된다.
각 위치에서 모델은 vocabulary에 있는 모든 문자에 대해 점수를 출력한다.
이 점수는 softmax를 통해 확률분포로 바뀔 수 있다.
모델은 다음에 올 문자 후보들에 대해 확률을 계산한다.
정답 문자와 모델의 예측을 비교하기 위해 cross entropy loss를 사용한다.
학습 과정에서는 AdamW optimizer를 사용하여 loss가 작아지도록 모델의 파라미터를 업데이트한다.

# 6. 글 생성 방식
학습이 끝난 모델은 시작 문장을 입력받아 새로운 글자를 생성할 수 있다.
생성 과정은 다음과 같다.

1. 시작 문장을 입력한다.
2. 모델이 다음 문자 확률분포를 계산한다.
3. 확률분포에서 다음 문자를 하나 샘플링한다.
4. 생성된 문자를 기존 문장 뒤에 붙인다.
5. 이 과정을 반복한다.


