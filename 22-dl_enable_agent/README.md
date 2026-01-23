# Enable Agent Deep Learning Tutorial

PyTorch 딥러닝 모델을 Enable Agent로 변환하고 RAG 챗봇으로 통합하는 튜토리얼이다.

## 개요

이 프로젝트는 Spaceship Titanic 데이터셋을 사용하여 승객의 다른 차원 이동 여부를 예측하는 딥러닝 모델을 Enable Agent로 변환한다.

### Enable Agent란?

기존 ML/DL 모델을 LLM 기반 에이전트 시스템에 통합하기 위한 래퍼 패턴이다.

```
+------------------------------------------+
|             Enable Agent                 |
+------------------------------------------+
|  Skill Definition (YAML)                 |
|  - 에이전트 이름, 설명                     |
|  - 입력/출력 스키마                        |
|  - 모델 정보                              |
+------------------------------------------+
|  PyTorch Model                           |
|  - SpaceshipClassifier (Neural Network)  |
|  - StandardScaler                        |
+------------------------------------------+
|  Agent Logic                             |
|  - 입력 검증                              |
|  - 예측 실행                              |
|  - 결과 구조화                            |
|  - AI 설명 생성                           |
+------------------------------------------+
```

## 디렉토리 구조

```
enable_agent_dl_tutorial/
├── models/                          # 학습된 모델 저장
│   ├── spaceship_classifier.pth     # PyTorch 모델
│   ├── spaceship_scaler.pkl         # StandardScaler
│   └── spaceship_classifier_metadata.json
├── skills/                          # Agent 스킬 정의
│   └── agent_skill.yaml
├── context_store/                   # 예측 결과 컨텍스트
│   ├── prediction_logs.json
│   ├── prediction_summary.json
│   └── knowledge_base.txt
├── src/                            # 소스 코드
│   ├── __init__.py
│   ├── train_model.py              # 모델 학습
│   ├── enable_agent.py             # Enable Agent 구현
│   ├── context_builder.py          # Context Builder
│   └── rag_chatbot.py              # RAG 챗봇
├── main.py                         # 메인 실행 스크립트
├── spaceship-preprocessing.csv     # 데이터셋
├── requirements.txt                # 패키지 의존성
└── README.md
```

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. OpenAI API 키 설정

`.env` 파일 생성:

```bash
OPENAI_API_KEY=your-api-key-here
```

## 사용법

### 전체 파이프라인 실행

```bash
python main.py
```

이 명령은 다음을 순차적으로 수행한다:
1. PyTorch 모델 학습
2. Enable Agent 설정
3. Context Builder 초기화
4. 데모 예측 실행
5. RAG 챗봇 시작

### 개별 단계 실행

```bash
# 모델 학습만
python main.py --train

# 챗봇만 실행 (학습된 모델 필요)
python main.py --chat

# 데모 예측만
python main.py --demo

# 챗봇 없이 실행
python main.py --no-chat
```

### 커스텀 데이터 사용

```bash
python main.py --data your_data.csv
```

## 모듈 설명

### train_model.py

PyTorch 신경망 분류 모델을 학습한다.

```python
from src.train_model import train_model

model, scaler, metadata = train_model('spaceship-preprocessing.csv')
```

**모델 아키텍처:**
- Input (10) → Linear(64) → ReLU → Dropout(0.3)
- → Linear(32) → ReLU → Dropout(0.3)
- → Linear(16) → ReLU
- → Linear(1) → Sigmoid

### enable_agent.py

PyTorch 모델을 Enable Agent로 래핑한다.

```python
from src.enable_agent import SpaceshipEnableAgent

agent = SpaceshipEnableAgent('skills/agent_skill.yaml')

# 예측 수행
result = agent.predict({
    'HomePlanet': 0,      # Earth
    'CryoSleep': 1,       # Yes
    'Destination': 2,     # 55 Cancri e
    'Age': 28.0,
    'VIP': 0,
    'RoomService': 0.0,
    'FoodCourt': 0.0,
    'ShoppingMall': 0.0,
    'Spa': 0.0,
    'VRDeck': 0.0
})

print(result)
# {
#     "predicted_class": "Transported",
#     "probability": 0.85,
#     "confidence": 0.85,
#     "decoded_features": {"HomePlanet": "Earth", ...},
#     ...
# }
```

### context_builder.py

예측 결과를 저장하고 지식 베이스를 구축한다.

```python
from src.context_builder import PredictionContextBuilder

builder = PredictionContextBuilder('context_store')

# 예측 결과 저장
builder.add_prediction(result)

# 통계 조회
summary = builder.get_summary()

# 지식 베이스 조회
knowledge = builder.get_knowledge_base_content()
```

### rag_chatbot.py

지식 베이스를 활용하는 대화형 챗봇이다.

```python
from src.rag_chatbot import SpaceshipRAGChatbot

chatbot = SpaceshipRAGChatbot(agent, context_builder)

# 대화
response = chatbot.chat("지구에서 온 25세 냉동 수면 승객의 이동 예측해줘")

# 보고서 생성
report = chatbot.generate_report()
```

## 챗봇 명령어

| 명령어 | 설명 |
|--------|------|
| `quit` / `exit` | 챗봇 종료 |
| `report` | 분석 보고서 생성 |
| `reset` | 대화 초기화 |

## 데이터셋 특성

| 특성 | 설명 | 인코딩 |
|------|------|--------|
| HomePlanet | 출발 행성 | 0: Earth, 1: Europa, 2: Mars |
| CryoSleep | 냉동 수면 | 0: No, 1: Yes |
| Destination | 목적지 | 0: TRAPPIST-1e, 1: PSO J318.5-22, 2: 55 Cancri e |
| Age | 나이 | 연속형 |
| VIP | VIP 서비스 | 0: No, 1: Yes |
| RoomService | 룸서비스 지출 | 연속형 |
| FoodCourt | 푸드코트 지출 | 연속형 |
| ShoppingMall | 쇼핑몰 지출 | 연속형 |
| Spa | 스파 지출 | 연속형 |
| VRDeck | VR 데크 지출 | 연속형 |
| **Transported** | **이동 여부 (Target)** | 0: No, 1: Yes |

## Enable Agent vs 기존 방식

### 기존 방식

```python
input_array = np.array([[0, 1, 2, 28, 0, 0, 0, 0, 0, 0]])
input_scaled = scaler.transform(input_array)
input_tensor = torch.FloatTensor(input_scaled)
prob = model(input_tensor).item()
# -> 0.85 (무슨 의미인지 알기 어려움)
```

### Enable Agent 방식

```python
result = agent.predict({
    'HomePlanet': 0, 'CryoSleep': 1, 'Destination': 2,
    'Age': 28, 'VIP': 0, ...
})
# -> {
#     "predicted_class": "Transported",
#     "probability": 0.85,
#     "confidence": 0.85,
#     "decoded_features": {"HomePlanet": "Earth", ...}
# }
```

### 비교

| 항목 | 기존 방식 | Enable Agent |
|------|----------|-------------|
| 입력 형식 | 배열 (순서 중요) | 딕셔너리 (명시적) |
| 전처리 | 수동 | 자동 |
| 유효성 검증 | 없음 | 자동 |
| 출력 | 숫자 | 구조화된 결과 |
| 해석 | 어려움 | 명확함 |
| LLM 통합 | 불가 | Function Calling 지원 |
| 설명 생성 | 불가 | AI 기반 설명 |

## 확장 가능성

- 다른 딥러닝 모델로 확장 (CNN, RNN, Transformer)
- 벡터 DB 통합 (Pinecone, Weaviate)
- 웹 애플리케이션 (FastAPI, Streamlit)
- 모니터링 시스템 (Prometheus, Grafana)
- 모델 A/B 테스팅

## 라이선스

MIT License
