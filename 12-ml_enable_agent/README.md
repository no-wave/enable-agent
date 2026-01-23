# Iris Classification Enable Agent

Random Forest 기반 붓꽃(Iris) 품종 분류 Enable Agent이다.

## 개요

꽃받침과 꽃잎의 특성(길이, 너비)을 입력받아 붓꽃 품종(Setosa, Versicolor, Virginica)을 분류한다.

## 클래스

| 품종 | 한글명 | 특징 |
|------|--------|------|
| Setosa | 부채붓꽃 | 꽃잎 작음 (< 2cm) |
| Versicolor | 버시컬러 붓꽃 | 중간 크기 (3-5cm) |
| Virginica | 버지니카 붓꽃 | 꽃잎 큼 (> 5cm) |

## 특성

| 특성 | 한글명 | 범위 |
|------|--------|------|
| sepal length | 꽃받침 길이 | 4.3 ~ 7.9 cm |
| sepal width | 꽃받침 너비 | 2.0 ~ 4.4 cm |
| petal length | 꽃잎 길이 | 1.0 ~ 6.9 cm |
| petal width | 꽃잎 너비 | 0.1 ~ 2.5 cm |

## 설치

```bash
pip install -r requirements.txt
```

## OpenAI API 키 설정

`.env` 파일을 생성하고 API 키를 설정한다:

```
OPENAI_API_KEY=your-api-key-here
```

## 사용법

### 전체 파이프라인 실행

```bash
python main.py
```

### 모델 학습만 실행

```bash
python main.py --train
```

### 챗봇만 실행 (학습된 모델 필요)

```bash
python main.py --chat
```

### 데모 예측 실행

```bash
python main.py --demo
```

## 디렉토리 구조

```
iris_enable_agent/
├── models/                    # 학습된 모델 저장
│   ├── iris_classifier.pkl    # Random Forest 모델
│   └── iris_classifier_metadata.json
├── skills/                    # Agent 스킬 정의
│   └── agent_skill.yaml
├── context_store/             # 예측 결과 컨텍스트
│   ├── prediction_logs.json
│   ├── prediction_summary.json
│   └── knowledge_base.txt
├── src/                       # 소스 코드
│   ├── __init__.py
│   ├── train_model.py         # 모델 학습
│   ├── enable_agent.py        # Enable Agent
│   ├── context_builder.py     # Context Builder
│   └── rag_chatbot.py         # RAG 챗봇
├── main.py                    # 메인 실행 스크립트
├── requirements.txt           # 패키지 의존성
└── README.md                  # 이 파일
```

## 챗봇 명령어

- `종료`: 'quit', 'exit'
- `보고서 생성`: 'report'
- `대화 초기화`: 'reset'

## 예시 대화

```
사용자: 꽃잎 길이 4.5cm, 너비 1.5cm, 꽃받침 길이 6cm, 너비 3cm인 붓꽃은?
챗봇: 입력된 특성을 분석한 결과, 이 붓꽃은 Versicolor(버시컬러 붓꽃)로 예측된다.
      신뢰도는 87%이다. 꽃잎 길이가 4.5cm로 중간 크기이고...

사용자: setosa 품종의 특징은?
챗봇: Setosa(부채붓꽃)는 다음과 같은 특징을 가진다:
      - 꽃잎이 작다 (길이 < 2cm, 너비 < 0.5cm)
      - 꽃받침이 상대적으로 넓다
      - 다른 품종과 명확하게 구분된다
```

## 모델 성능

- 알고리즘: Random Forest Classifier
- 정확도: ~93%
- 5-Fold 교차 검증: ~95%

## 특성 중요도

모델에서 가장 중요한 특성:
1. petal width (꽃잎 너비): ~43%
2. petal length (꽃잎 길이): ~43%
3. sepal length (꽃받침 길이): ~12%
4. sepal width (꽃받침 너비): ~1%

## 라이선스

MIT License
