#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enable Agent Deep Learning Tutorial - 메인 실행 스크립트

이 스크립트는 전체 Enable Agent 파이프라인을 실행한다:
1. PyTorch 딥러닝 모델 학습
2. Enable Agent 초기화
3. Context Builder 설정
4. RAG 챗봇 실행

사용법:
    python main.py                  # 전체 파이프라인 실행
    python main.py --train          # 모델 학습만 실행
    python main.py --chat           # 챗봇만 실행 (학습된 모델 필요)
    python main.py --demo           # 데모 예측 실행
"""

import argparse
import json
import sys
import io
from pathlib import Path

# 인코딩 강제 설정 (import 직후에)
try:
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


def train_model_step(data_path: str):
    """모델 학습 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 1: PyTorch 딥러닝 모델 학습")
    print("=" * 60)
    
    from src.train_model import train_model
    model, scaler, metadata = train_model(data_path)
    
    return model, scaler, metadata


def setup_agent_step():
    """Enable Agent 설정 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 2: Enable Agent 설정")
    print("=" * 60)
    
    from src.enable_agent import create_skill_file, SpaceshipEnableAgent
    
    # 스킬 파일 생성
    skill_path = create_skill_file('skills/agent_skill.yaml')
    
    # Agent 초기화
    agent = SpaceshipEnableAgent(skill_path)
    
    return agent


def setup_context_builder_step():
    """Context Builder 설정 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 3: Context Builder 설정")
    print("=" * 60)
    
    from src.context_builder import PredictionContextBuilder
    
    context_builder = PredictionContextBuilder('context_store')
    
    return context_builder


def run_demo_predictions(agent, context_builder):
    """데모 예측을 실행한다."""
    print("\n" + "=" * 60)
    print("Step 4: 데모 예측 실행")
    print("=" * 60)
    
    test_samples = [
        # 냉동 수면 승객 (이동 가능성 높음)
        {
            'HomePlanet': 0, 'CryoSleep': 1, 'Destination': 2,
            'Age': 25.0, 'VIP': 0,
            'RoomService': 0.0, 'FoodCourt': 0.0, 'ShoppingMall': 0.0,
            'Spa': 0.0, 'VRDeck': 0.0
        },
        # 고지출 VIP 승객 (이동 가능성 낮음)
        {
            'HomePlanet': 1, 'CryoSleep': 0, 'Destination': 0,
            'Age': 45.0, 'VIP': 1,
            'RoomService': 2000.0, 'FoodCourt': 3000.0, 'ShoppingMall': 1000.0,
            'Spa': 4000.0, 'VRDeck': 1500.0
        },
        # 저지출 비냉동 승객
        {
            'HomePlanet': 2, 'CryoSleep': 0, 'Destination': 1,
            'Age': 30.0, 'VIP': 0,
            'RoomService': 100.0, 'FoodCourt': 200.0, 'ShoppingMall': 50.0,
            'Spa': 0.0, 'VRDeck': 300.0
        },
    ]
    
    print("\n=== 예측 결과 ===\n")
    
    for i, sample in enumerate(test_samples, 1):
        result = agent.predict(sample)
        context_builder.add_prediction(result)
        
        print(f"[예측 {i}]")
        print(f"  입력: HomePlanet={sample['HomePlanet']}, CryoSleep={sample['CryoSleep']}, Age={sample['Age']}")
        print(f"  결과: {result['predicted_class']}")
        print(f"  확률: {result['probability']:.2%}")
        print(f"  신뢰도: {result['confidence']:.2%}")
        print()
    
    # 통계 출력
    summary = context_builder.get_summary()
    print("=== 예측 통계 ===")
    print(f"총 예측: {summary['total_predictions']}회")
    print(f"평균 신뢰도: {summary['average_confidence']:.2%}")
    print(f"이동 예측: {summary['class_distribution'].get('Transported', 0)}회")
    print(f"비이동 예측: {summary['class_distribution'].get('Not Transported', 0)}회")


def run_chatbot(agent, context_builder):
    """RAG 챗봇을 실행한다."""
    print("\n" + "=" * 60)
    print("Step 5: RAG 챗봇 실행")
    print("=" * 60)
    
    from src.rag_chatbot import SpaceshipRAGChatbot
    
    chatbot = SpaceshipRAGChatbot(agent, context_builder)
    
    print("\n" + "=" * 60)
    print("Spaceship Titanic RAG 챗봇")
    print("=" * 60)
    print("종료: 'quit' 또는 'exit'")
    print("보고서: 'report'")
    print("초기화: 'reset'")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("사용자: ").strip()
        except (UnicodeDecodeError, UnicodeError):
            print("[WARNING] 입력 인코딩 오류, 다시 입력해달라.")
            continue
        except (EOFError, KeyboardInterrupt):
            print("\n챗봇을 종료한다.")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("챗봇을 종료한다.")
            break
        
        if user_input.lower() == 'report':
            print("\n보고서 생성 중...")
            report = chatbot.generate_report()
            print("\n" + "=" * 60)
            print("분석 보고서")
            print("=" * 60)
            print(report)
            print("=" * 60 + "\n")
            
            # 보고서 저장
            report_path = Path('context_store/analysis_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"보고서 저장: {report_path}\n")
            continue
        
        if user_input.lower() == 'reset':
            chatbot.reset_conversation()
            continue
        
        try:
            response = chatbot.chat(user_input)
            print(f"\n챗봇: {response}\n")
        except Exception as e:
            print(f"\n[ERROR] 응답 생성 실패: {e}\n")
    
    return chatbot


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='Enable Agent Deep Learning Tutorial',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python main.py                          # 전체 파이프라인 실행
    python main.py --train                  # 모델 학습만
    python main.py --chat                   # 챗봇만 실행
    python main.py --demo                   # 데모 예측만
    python main.py --data custom_data.csv   # 커스텀 데이터 사용
        """
    )
    
    parser.add_argument('--train', action='store_true', help='모델 학습만 실행')
    parser.add_argument('--chat', action='store_true', help='챗봇만 실행')
    parser.add_argument('--demo', action='store_true', help='데모 예측만 실행')
    parser.add_argument('--data', type=str, default='spaceship-preprocessing.csv', help='데이터 파일 경로')
    parser.add_argument('--no-chat', action='store_true', help='챗봇 실행 건너뛰기')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Enable Agent Deep Learning Tutorial")
    print("PyTorch 기반 딥러닝 모델을 Enable Agent로 변환")
    print("=" * 60)
    
    # 모델 학습만 실행
    if args.train:
        train_model_step(args.data)
        return
    
    # 챗봇만 실행 (학습된 모델 필요)
    if args.chat:
        model_path = Path('models/spaceship_classifier.pth')
        if not model_path.exists():
            print("오류: 학습된 모델이 없다. 먼저 --train으로 모델을 학습하라.")
            sys.exit(1)
        
        agent = setup_agent_step()
        context_builder = setup_context_builder_step()
        run_chatbot(agent, context_builder)
        return
    
    # 데모 예측만 실행
    if args.demo:
        model_path = Path('models/spaceship_classifier.pth')
        if not model_path.exists():
            print("오류: 학습된 모델이 없다. 먼저 --train으로 모델을 학습하라.")
            sys.exit(1)
        
        agent = setup_agent_step()
        context_builder = setup_context_builder_step()
        run_demo_predictions(agent, context_builder)
        return
    
    # 전체 파이프라인 실행
    model_path = Path('models/spaceship_classifier.pth')
    
    # Step 1: 모델 학습 (이미 학습된 모델이 없는 경우)
    if not model_path.exists():
        train_model_step(args.data)
    else:
        print("\n기존 학습된 모델을 사용한다.")
        print(f"모델 경로: {model_path}")
    
    # Step 2: Enable Agent 설정
    agent = setup_agent_step()
    
    # Step 3: Context Builder 설정
    context_builder = setup_context_builder_step()
    
    # Step 4: 데모 예측
    run_demo_predictions(agent, context_builder)
    
    # Step 5: 챗봇 실행 (옵션)
    if not args.no_chat:
        print("\n챗봇을 시작하려면 Enter를 누르세요. 건너뛰려면 'skip'을 입력하세요.")
        try:
            user_choice = input().strip().lower()
        except (UnicodeDecodeError, EOFError):
            user_choice = 'skip'
        
        if user_choice != 'skip':
            run_chatbot(agent, context_builder)
    
    print("\n" + "=" * 60)
    print("Enable Agent 튜토리얼 완료")
    print("=" * 60)
    
    # 최종 통계 출력
    summary = context_builder.get_summary()
    if summary:
        print(f"\n총 예측 수행: {summary['total_predictions']}회")
        print(f"생성된 파일:")
        print(f"  - models/spaceship_classifier.pth")
        print(f"  - models/spaceship_scaler.pkl")
        print(f"  - models/spaceship_classifier_metadata.json")
        print(f"  - skills/agent_skill.yaml")
        print(f"  - context_store/prediction_logs.json")
        print(f"  - context_store/prediction_summary.json")
        print(f"  - context_store/knowledge_base.txt")


if __name__ == '__main__':
    main()
