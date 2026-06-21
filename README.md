# GPT 2.0: 문자 단위 GPT 구현 프로젝트

## 1. 프로젝트 소개

이 프로젝트는 GPT의 핵심 원리를 작은 규모로 직접 구현한 문자 단위 언어 모델 프로젝트이다.

모델은 `input.txt`에 들어 있는 글을 학습한 뒤, 앞에 나온 문자들을 바탕으로 다음 문자를 예측한다. 일반적인 GPT 모델은 단어나 subword 단위의 토큰을 사용하지만, 이 프로젝트에서는 구현을 단순화하기 위해 문자 하나하나를 토큰으로 사용하였다.

이 프로젝트는 실제 GPT-2 전체를 그대로 복제한 것은 아니다. 대신 GPT 계열 모델의 핵심 구조인 다음 토큰 예측, token embedding, position embedding, masked self-attention, multi-head attention, feedforward network, residual connection, layer normalization을 작은 Transformer 모델로 구현한 것이다.

## 2. 프로젝트 목표

이 프로젝트의 목표는 다음과 같다.

1. 텍스트 데이터를 문자 단위로 토큰화한다.
2. 각 문자를 정수 id로 변환한다.
3. 입력 sequence를 이용하여 다음 문자를 예측하는 모델을 만든다.
4. Transformer decoder 기반의 작은 GPT 모델을 구현한다.
5. 학습된 모델을 이용하여 새로운 글을 생성한다.
6. GPT 계열 언어 모델의 기본 작동 원리를 이해한다.

## 3. 전체 시스템 구조

이 프로젝트의 전체 흐름은 다음과 같다.

```text
input.txt 읽기
→ 문자 단위 vocabulary 생성
→ 문자를 정수 id로 변환
→ 학습 데이터와 검증 데이터 분리
→ 입력 x와 정답 y batch 생성
→ TinyGPT 모델 생성
→ 다음 문자 예측 학습
→ loss 계산
→ backpropagation
→ 모델 파라미터 업데이트
→ 새로운 글 생성
→ 학습된 모델 저장
```

## 4. 학습 데이터 구조

모델은 텍스트를 문자 단위로 읽는다.

예를 들어 다음 문장이 있다고 하자.

```text
나는 오늘 밥을 먹었다
```

이 문장은 다음과 같은 문자들의 sequence로 처리된다.

```text
나, 는,  , 오, 늘,  , 밥, 을,  , 먹, 었, 다
```

모델은 현재까지의 문자들을 보고 다음 문자를 맞히도록 학습된다.

예를 들어 입력과 정답은 다음과 같이 한 칸씩 밀린 형태가 된다.

```text
입력 x: 나는 오늘 밥을 먹었
정답 y: 는 오늘 밥을 먹었다
```

즉, 모델은 `나` 다음에 `는`, `는` 다음에 공백, 공백 다음에 `오`가 오는 식의 패턴을 학습한다.

## 5. 문자 단위 토큰화

이 프로젝트에서는 단어가 아니라 문자 하나하나를 토큰으로 사용한다.

예를 들어 다음 문장이 있다고 하자.

```text
나는 GPT를 만든다
```

이 문장은 다음과 같은 문자 토큰들로 나뉜다.

```text
나, 는,  , G, P, T, 를,  , 만, 든, 다
```

각 문자는 vocabulary에 등록되고 고유한 정수 id를 가진다.

예시는 다음과 같다.

```text
나 → 0
는 → 1
G → 2
P → 3
T → 4
```

모델은 문자를 직접 입력받는 것이 아니라, 각 문자를 정수 id로 바꾼 값을 입력으로 사용한다.

## 6. 모델 구조

구현한 TinyGPT 모델은 다음 요소들로 구성된다.

1. Token Embedding
2. Position Embedding
3. Masked Self-Attention
4. Multi-Head Attention
5. FeedForward Network
6. Residual Connection
7. Layer Normalization
8. Linear Output Layer

전체 구조는 다음과 같다.

```text
문자 id 입력
→ token embedding
→ position embedding 더하기
→ Transformer Block 반복
→ LayerNorm
→ Linear Layer
→ 다음 문자 확률 예측
```

## 7. Token Embedding

문자 id는 그 자체로는 의미 있는 벡터가 아니다.

따라서 모델은 embedding layer를 사용하여 각 문자 id를 학습 가능한 벡터로 변환한다.

```text
문자 id → token embedding vector
```

예를 들어 `나`라는 문자가 정수 id `0`으로 표현된다면, 모델은 이 id를 일정한 차원의 벡터로 바꾼다.

이 벡터는 학습 과정에서 계속 업데이트되며, 문자들의 관계와 패턴을 표현하게 된다.

## 8. Position Embedding

Transformer는 RNN처럼 입력을 순서대로 처리하지 않는다.  
따라서 각 문자가 문장 안에서 몇 번째 위치에 있는지 알려주는 정보가 필요하다.

이를 위해 position embedding을 사용한다.

```text
최종 입력 벡터 = token embedding + position embedding
```

token embedding이 “어떤 문자냐”를 표현한다면, position embedding은 “그 문자가 몇 번째 위치에 있느냐”를 표현한다.

이 두 정보를 더함으로써 모델은 문자 자체의 정보와 문장 안에서의 위치 정보를 함께 사용할 수 있다.

## 9. Masked Self-Attention

Self-attention은 각 문자가 문장 안의 다른 문자들을 얼마나 참고할지 계산하는 구조이다.

하지만 GPT는 다음 문자를 예측하는 모델이기 때문에 현재 위치에서 미래의 문자를 보면 안 된다.

예를 들어 다음 문장을 학습한다고 하자.

```text
나는 밥을 먹었다
```

`먹`을 예측하는 시점에서 모델이 이미 뒤에 있는 `었`, `다`를 보면 정답을 미리 보는 것이 된다.

따라서 이 프로젝트에서는 lower triangular mask를 사용하여 미래 위치를 가린다.

```text
현재 위치에서는 자기 자신과 과거 위치만 참고 가능
미래 위치는 참고 불가능
```

이 구조를 masked self-attention이라고 한다.

## 10. Multi-Head Attention

하나의 attention head만 사용하면 문맥을 한 가지 관점에서만 바라보게 된다.

Multi-head attention은 여러 개의 attention head를 병렬로 사용하여 다양한 관점에서 문맥 정보를 학습한다.

예를 들어 어떤 head는 가까운 문자를 중요하게 볼 수 있고, 다른 head는 문장 전체의 흐름을 중요하게 볼 수 있다.

이 프로젝트에서는 여러 attention head의 출력을 이어붙인 뒤, 다시 linear layer를 통과시켜 최종 attention 결과를 만든다.

## 11. FeedForward Network

Attention이 문자들 사이의 관계를 학습하는 부분이라면, feedforward network는 각 위치의 벡터 표현을 더 깊게 변환하는 부분이다.

이 프로젝트의 feedforward network 구조는 다음과 같다.

```text
Linear
→ ReLU
→ Linear
→ Dropout
```

이를 통해 모델은 attention으로 얻은 문맥 정보를 더 복잡한 표현으로 바꿀 수 있다.

## 11. Residual Connection과 Layer Normalization

Transformer block에서는 residual connection과 layer normalization을 사용한다.

Residual connection은 입력 x에 attention 결과나 feedforward 결과를 더해주는 구조이다.

```text
x = x + attention(x)
x = x + feedforward(x)
```

이 구조를 사용하면 깊은 신경망에서도 정보와 gradient가 더 잘 흐를 수 있다.

Layer normalization은 각 층의 입력 분포를 안정화하는 역할을 한다.  
이 프로젝트에서는 attention과 feedforward network를 적용하기 전에 layer normalization을 사용하였다.

```text
x = x + attention(layer_norm(x))
x = x + feedforward(layer_norm(x))
```

이 구조는 학습을 안정적으로 만들어준다.

## 12. 학습 방식

모델은 다음 문자 예측 문제를 풀도록 학습된다.

각 위치에서 모델은 vocabulary에 있는 모든 문자에 대해 점수를 출력한다.  
이 점수는 softmax를 통해 확률분포로 바뀔 수 있다.

예를 들어 현재 문맥이 다음과 같다고 하자.

```text
나는 밥을
```

모델은 다음에 올 문자 후보들에 대해 확률을 계산한다.

```text
" "일 확률
"먹"일 확률
"마"일 확률
"갔"일 확률
...
```

정답 문자와 모델의 예측을 비교하기 위해 cross entropy loss를 사용한다.  
학습 과정에서는 AdamW optimizer를 사용하여 loss가 작아지도록 모델의 파라미터를 업데이트한다.

## 13. 글 생성 방식

학습이 끝난 모델은 시작 문장을 입력받아 새로운 글자를 생성할 수 있다.

생성 과정은 다음과 같다.

```text
1. 시작 문장을 입력한다.
2. 모델이 다음 문자 확률분포를 계산한다.
3. 확률분포에서 다음 문자를 하나 샘플링한다.
4. 생성된 문자를 기존 문장 뒤에 붙인다.
5. 이 과정을 반복한다.
```

예를 들어 시작 문장이 다음과 같다면,

```text
나는
```

모델은 다음 문자를 하나 생성하고, 다시 그 결과를 입력으로 사용하여 다음 문자를 생성한다.

이 과정을 반복하면 새로운 문장이 만들어진다.
