# YOLO Plastic Detection Enable Agent

YOLOv8 기반 플라스틱 재활용 분류 객체 탐지 Enable Agent이다.

## 개요

이미지에서 플라스틱 객체(PET, PS, PP, PE)를 탐지하고 재활용 가이드를 제공한다.

## 클래스

| 클래스 | 설명 | 재활용 코드 | 재활용 가능 |
|--------|------|-------------|-------------|
| PET | 폴리에틸렌 테레프탈레이트 | #1 | ✅ |
| PS | 폴리스타이렌 | #6 | ❌ |
| PP | 폴리프로필렌 | #5 | ✅ |
| PE | 폴리에틸렌 | #2, #4 | ✅ |

## 설치

```bash
pip install -r requirements.txt
```

## OpenAI API 키 설정

`.env` 파일을 생성하고 API 키를 설정한다:

```
OPENAI_API_KEY=your-api-key-here
```

## 데이터 준비

`data/raw_data` 디렉토리에 다음 구조로 데이터를 넣는다:

```
data/
└── raw_data/
    ├── annotations/    # COCO JSON 파일들
    │   ├── image1.json
    │   ├── image2.json
    │   └── ...
    └── images/         # 이미지 파일들
        ├── image1.jpg
        ├── image2.jpg
        └── ...
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

### 데모 탐지 실행

```bash
python main.py --demo
```

## 디렉토리 구조

```
yolo_plastic_enable_agent/
├── models/                    # 학습된 모델 저장
│   ├── plastic_yolov8_best.pt # YOLOv8 모델
│   └── yolo_metadata.json     # 메타데이터
├── skills/                    # Agent 스킬 정의
│   └── agent_skill.yaml
├── context_store/             # 탐지 결과 컨텍스트
│   ├── detection_logs.json
│   ├── detection_summary.json
│   └── knowledge_base.txt
├── datasets/                  # YOLO 포맷 데이터셋 (자동 생성)
│   ├── plastic.yaml
│   └── plastic/
│       ├── train/
│       └── val/
├── data/                      # 원본 데이터
│   └── raw_data/
│       ├── annotations/
│       └── images/
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
- `이미지 분석`: 'analyze <이미지경로>'
- `대화 초기화`: 'reset'

## 예시 대화

```
사용자: PET 플라스틱은 어떻게 버려야 해?
챗봇: PET(폴리에틸렌 테레프탈레이트)는 재활용이 가능한 플라스틱이다. 
      라벨을 제거하고 내용물을 비운 후 압축하여 플라스틱류로 배출하면 된다.
      주로 음료수 병, 생수 병에 사용된다.

사용자: analyze test_image.jpg
챗봇: 이미지에서 3개의 플라스틱 객체를 탐지했다.
      - PET: 2개 (재활용 가능)
      - PS: 1개 (재활용 불가)
      ...
```

## 라이선스

MIT License
