# Enable Agent로 똑똑한 AI 에이전트 만들기
기존 ML/DL 모델을 대화형 AI 에이전트로 Enable 전환 실전 가이드


<img src="https://beat-by-wire.gitbook.io/beat-by-wire/~gitbook/image?url=https%3A%2F%2F3055094660-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FYzxz4QeW9UTrhrpWwKiQ%252Fuploads%252FypgroGjhlTxVelbYRQey%252FFrame%25207.png%3Falt%3Dmedia%26token%3D3ba304e0-8cb6-47f4-bbc6-69eac7a68cfc&width=300&dpr=3&quality=100&sign=cb064d68&sv=2" width="500" height="707"/>

## 책 소개

인공지능 분야에서 머신러닝과 딥러닝 모델은 이미 수많은 문제를 해결하며 그 가치를 입증했다. 이미지 분류, 객체 탐지, 시계열 예측, 이상치 탐지 등 다양한 도메인에서 전문화된 모델들이 개발되었고, 많은 조직이 이를 실무에 활용하고 있다. 그러나 이러한 모델들을 실제 서비스에 통합하고 운영하는 과정은 여전히 기술적 장벽이 높다. 데이터 과학자가 Jupyter 노트북에서 95% 정확도를 달성한 모델이 실제 사용자에게 가치를 전달하기까지는 넘어야 할 산이 많다.

전통적인 머신러닝 모델 배포 방식은 여러 근본적인 한계를 가지고 있다. 첫째, 모델을 사용하기 위해서는 프로그래밍 지식이 필수적이다. numpy 배열을 준비하고, API를 호출하며, 반환된 숫자 코드를 해석하는 과정은 기술 전문가에게는 자연스럽지만, 영업 관리자나 마케팅 담당자에게는 접근조차 어렵다. 둘째, 각 모델이 서로 다른 인터페이스와 요구사항을 가지고 있어 통합과 유지보수가 복잡하다. YOLO 객체 탐지 모델과 LSTM 시계열 예측 모델을 하나의 시스템에 통합하려면 각각에 대한 별도의 전처리, 호출, 후처리 로직을 구현해야 한다. 셋째, 모델의 예측 결과가 일회성으로 소비되고 사라진다. 과거에 어떤 예측을 했는지, 성능이 어떻게 변화했는지, 어떤 패턴이 있는지 체계적으로 추적하고 활용하기 어렵다.

이러한 배경에서 대규모 언어 모델(LLM)의 등장은 새로운 가능성을 열었다. GPT, Claude와 같은 모델들은 자연어로 대화하며 복잡한 작업을 수행할 수 있다. 그러나 LLM만으로는 전문화된 도메인 문제를 해결하기 어렵다. 범용 LLM은 의료 영상 분석, 금융 사기 탐지, 제조 품질 검사 같은 특수한 작업에서 전문 모델의 정확도를 따라잡기 힘들다. 진정한 혁신은 LLM의 자연어 이해 능력과 기존 전문 모델의 도메인 전문성을 결합하는 데 있다.

Enable Agent는 바로 이 지점에서 출발한다. Enable Agent는 기존의 머신러닝, 딥러닝 모델을 LLM 기반 대화형 에이전트로 전환하는 설계 패턴이자 방법론이다. "Enable"이라는 이름은 기존 모델을 단순히 감싸는 것이 아니라, 자연어 인터페이스, 설명 가능성, 컨텍스트 관리, 지속적 개선 능력을 '가능하게 만든다(Enable)'는 의미를 담고 있다. 전문화된 AI 모델을 누구나 대화하듯 활용할 수 있고, 과거 경험에서 학습하며, 스스로 개선되는 지능형 에이전트로 진화시키는 것이 Enable Agent의 핵심이다.

본 책은 Enable Agent를 이론적 개념 수준에서 다루는 것이 아니라, 실무에서 즉시 구현하고 활용할 수 있는 실용적인 가이드를 제공한다. 네 가지 실제 도메인(머신러닝 분류, 딥러닝 예측, 금융 사기 탐지, 컴퓨터 비전 객체 탐지)을 예제로 사용하여, 각 도메인의 특성에 맞는 Enable Agent 구축 방법을 단계별로 설명한다.

본 서의 목표는 단순히 기술을 전달하는 것을 넘어선다. Enable Agent를 통해 AI를 민주화하고, 조직 내 모든 구성원이 전문화된 AI 모델의 도움을 받을 수 있게 만드는 것이 궁극적인 목표다. 데이터 과학자가 개발한 모델을 영업 관리자가 자연어로 활용하고, 마케팅 팀이 챗봇을 통해 수요 예측을 얻으며, 재활용 센터 직원이 스마트폰으로 플라스틱 종류를 즉시 판별하는 미래를 현실로 만드는 것이다.

또한 본 서는 단발성 프로젝트가 아닌, 지속적으로 개선되는 AI 시스템 구축 방법을 제시한다. Context Builder를 통해 모든 예측을 자동으로 기록하고, 성능 변화를 추적하며, 사용자 피드백을 수집하고, 데이터 기반 재학습 의사결정을 내리는 완전한 라이프사이클을 구현한다. 한 번 배포하고 잊어버리는 모델이 아니라, 사용될수록 똑똑해지고 조직의 자산이 되는 에이전트를 만드는 방법을 배운다.

Enable Agent는 더 이상 소수의 전문가만이 다루는 고급 기술이 아니다. 적절한 도구와 지식을 갖춘다면 누구나 자신의 조직에서 강력한 AI 에이전트 시스템을 구축할 수 있다. 본 서가 기존 AI 모델을 똑똑한 에이전트로 전환하고, 조직 전체에 AI를 내재화하려는 모든 개발자, 데이터 과학자, ML 엔지니어, 그리고 AI 활용을 고민하는 실무자에게 실용적인 가이드가 되기를 바란다.



## 목 차

저자 소개
Table of Contents (목차)
서문: 들어가며

제1부: Enable Agent, 에이전트 전환
1.1. Enable Agent 개요
1.2. Enable Agent의 이점
1.3. 지속적 개선과 피드백
1.4. Enable Agent 패턴 방식

제2부: Enable Agent: 머신 러닝
Part 1. Enable Agent: ML 모델 학습
Part 2. Enable Agent: ML 모델의 Enable Agent화
Part 3. Enable Agent: 예측 결과 컨텍스트화
Part 4. Enable Agent: RAG 챗봇 및 보고서 생성
Part 5. Enable Agent 프로젝트

제3부: Enable Agent 딥러닝
Part 1. Enable Agent: PyTorch 딥러닝 모델 학습
Part 2. Enable Agent: Enable Agent 구현
Part 3. Enable Agent: Context Builder 구현
Part 4. Enable Agent: RAG 챗봇 구현
Part 5. Enable Agent 프로젝트

제4부: Enable Agent FDS 이상치 탐지
Part 1. Enable Agent: 이상치 탐지 모델 학습 (FDS)
Part 2. Enable Agent: FDS Enable Agent 구현
Part 3. Enable Agent: FDS Context Builder
Part 4. Enable Agent: FDS RAG 챗봇
Part 5. Enable Agent 프로젝트

제5부: Enable Agent: Vision 객체 탐지
Part 1. Enable Agent: YOLOv8 객체 탐지 모델 학습
Part 2. Enable Agent: YOLO Enable Agent 구현
Part 3. Enable Agent: Detection Context Builder 구현
Part 4. Enable Agent: Plastic Detection RAG 챗봇
Part 5. Enable Agent 프로젝트

결론: 마무리 하며
References: 참고 문헌


## E-Book 구매

- Yes24: https://www.yes24.com/product/goods/176018161
- 교보문고: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000012516536
- 알라딘: http://aladin.kr/p/YCsMW

## Github 코드: 

https://github.com/no-wave/enable-agent



