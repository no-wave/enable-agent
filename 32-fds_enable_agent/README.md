# FDS Enable Agent

이상치 탐지 기반 사기 탐지 시스템(FDS)을 Enable Agent로 구현한 프로젝트다.

## 개요

정상 거래 패턴만 학습하여 새로운 유형의 사기에도 대응할 수 있는 비지도 학습 방식의 사기 탐지 시스템이다.

## 특징

### 이상치 탐지 방식

| 항목 | 지도 학습 | 이상치 탐지 (본 프로젝트) |
|------|----------|----------------------|
| 학습 데이터 | 정상 + 사기 | 정상만 사용 |
| 새로운 사기 패턴 | 재학습 필요 | 자동 탐지 |
| 클래스 불균형 | 문제 발생 | 영향 없음 |

### 앙상블 모델

- **Isolation Forest**: 이상치를 빠르게 고립시켜 탐지
- **Autoencoder**: 복원 오차 기반 이상치 탐지
- 두 모델의 점수를 결합하여 더 강건한 탐지

## 디렉토리 구조

```
fds_enable_agent/
├── models/                          # 학습된 모델 (자동 생성)
│   ├── isolation_forest.pkl
│   ├── autoencoder.pth
│   ├── fds_scaler.pkl
│   ├── label_encoders.pkl
│   └── fds_metadata.json
├── skills/                          # Agent 스킬 정의 (자동 생성)
│   └── agent_skill.yaml
├── context_store/                   # 분석 결과 저장 (자동 생성)
│   ├── analysis_logs.json
│   ├── fds_summary.json
│   └── fds_knowledge_base.txt
├── src/
│   ├── __init__.py
│   ├── train_model.py              # 이상치 탐지 모델 학습
│   ├── enable_agent.py             # FDS Enable Agent 구현
│   ├── context_builder.py          # Context Builder
│   └── rag_chatbot.py              # RAG 챗봇
├── main.py                         # 전체 실행 스크립트
├── Fraud_Data.csv                  # 데이터셋
├── requirements.txt
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## OpenAI API 키 설정

`.env` 파일 생성:

```
OPENAI_API_KEY=your-api-key-here
```

## 사용법

### 전체 파이프라인 실행

```bash
python main.py
```

### 개별 단계 실행

```bash
# 모델 학습만
python main.py --train

# 챗봇만 실행 (학습된 모델 필요)
python main.py --chat

# 데모 분석만
python main.py --demo

# 챗봇 없이 실행
python main.py --no-chat
```

### 커스텀 데이터 사용

```bash
python main.py --data your_data.csv
```

## 위험도 등급 시스템

| 등급 | 점수 범위 | 권장 조치 |
|------|----------|----------|
| LOW | 0 ~ 0.3 | 정상 처리 |
| MEDIUM | 0.3 ~ 0.5 | 모니터링 강화 |
| HIGH | 0.5 ~ 0.7 | 추가 인증 요청 |
| CRITICAL | 0.7 ~ 1.0 | 거래 보류, 수동 검토 |

## 챗봇 명령어

| 명령어 | 설명 |
|--------|------|
| `quit` / `exit` | 챗봇 종료 |
| `report` | 분석 보고서 생성 |
| `reset` | 대화 초기화 |

## 데이터셋 특성

| 특성 | 설명 |
|------|------|
| purchase_value | 구매 금액 |
| age | 사용자 나이 |
| source | 유입 경로 (SEO/Ads/Direct) |
| browser | 브라우저 |
| sex | 성별 |
| signup_time | 가입 시간 |
| purchase_time | 구매 시간 |
| class | 사기 여부 (0: 정상, 1: 사기) |

## 아키텍처

```
+------------------------------------------+
|           FDS RAG Chatbot                |
|  - 자연어 대화                            |
|  - 지식 베이스 검색                       |
|  - 보고서 생성                            |
+----+-------------------------------+-----+
     |                               |
     v                               v
+------------------+    +---------------------+
| FDS Enable Agent |    | FDS Context Builder |
| - Isolation Forest|   | - 로그 저장          |
| - Autoencoder     |   | - 통계 생성          |
| - 위험도 산정      |   | - 지식 베이스        |
+------------------+    +---------------------+
```

## 라이선스

MIT License
