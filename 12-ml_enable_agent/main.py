#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iris Classification Enable Agent - 메인 실행 스크립트

이 스크립트는 전체 Enable Agent 파이프라인을 실행한다:
1. Random Forest 모델 학습
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
from datetime import datetime

# 인코딩 강제 설정
try:
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()


def train_model_step():
    """모델 학습 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 1: Random Forest 모델 학습")
    print("=" * 60)
    
    from src.train_model import train_model
    
    model, accuracy, metadata = train_model()
    
    return model, accuracy, metadata


def setup_agent_step():
    """Enable Agent 설정 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 2: Enable Agent 설정")
    print("=" * 60)
    
    from src.enable_agent import IrisClassificationAgent, create_skill_file
    
    # 스킬 파일 생성
    skill_path = create_skill_file('skills/agent_skill.yaml')
    
    # Agent 초기화
    agent = IrisClassificationAgent(skill_path)
    
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
        # Setosa (작은 꽃잎)
        {
            'sepal_length': 5.1, 'sepal_width': 3.5,
            'petal_length': 1.4, 'petal_width': 0.2
        },
        # Versicolor (중간 크기)
        {
            'sepal_length': 6.0, 'sepal_width': 2.9,
            'petal_length': 4.5, 'petal_width': 1.5
        },
        # Virginica (큰 꽃잎)
        {
            'sepal_length': 6.7, 'sepal_width': 3.0,
            'petal_length': 5.2, 'petal_width': 2.3
        },
        # 경계 케이스
        {
            'sepal_length': 5.9, 'sepal_width': 3.0,
            'petal_length': 4.2, 'petal_width': 1.5
        },
    ]
    
    print("\n=== 예측 결과 ===\n")
    
    for i, sample in enumerate(test_samples, 1):
        result = agent.predict(sample)
        context_builder.add_prediction(result, metadata={'demo': True, 'index': i})
        
        print(f"[예측 {i}]")
        print(f"  입력: 꽃잎 {sample['petal_length']}x{sample['petal_width']}cm, "
              f"꽃받침 {sample['sepal_length']}x{sample['sepal_width']}cm")
        print(f"  결과: {result['predicted_class']} ({result['korean_name']})")
        print(f"  신뢰도: {result['confidence']:.1%}")
        print()
    
    # 통계 출력
    summary = context_builder.get_summary()
    print("=== 예측 통계 ===")
    print(f"총 예측: {summary['total_predictions']}회")
    print(f"평균 신뢰도: {summary['average_confidence']:.1%}")
    print(f"클래스별 분포: {summary['class_distribution']}")


def run_chatbot(agent, context_builder):
    """RAG 챗봇을 실행한다."""
    print("\n" + "=" * 60)
    print("Step 5: RAG 챗봇 실행")
    print("=" * 60)
    
    from src.rag_chatbot import interactive_chat
    
    return interactive_chat(agent, context_builder)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='Iris Classification Enable Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python main.py                  # 전체 파이프라인 실행
    python main.py --train          # 모델 학습만
    python main.py --chat           # 챗봇만 실행
    python main.py --demo           # 데모 예측만
        """
    )
    
    parser.add_argument('--train', action='store_true', help='모델 학습만 실행')
    parser.add_argument('--chat', action='store_true', help='챗봇만 실행')
    parser.add_argument('--demo', action='store_true', help='데모 예측만 실행')
    parser.add_argument('--no-chat', action='store_true', help='챗봇 실행 건너뛰기')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Iris Classification Enable Agent")
    print("붓꽃 품종 분류 시스템")
    print("=" * 60)
    
    # 모델 학습만 실행
    if args.train:
        train_model_step()
        return
    
    # 챗봇만 실행 (학습된 모델 필요)
    if args.chat:
        model_path = Path('models/iris_classifier.pkl')
        if not model_path.exists():
            print("오류: 학습된 모델이 없다. 먼저 --train으로 모델을 학습하라.")
            sys.exit(1)
        
        agent = setup_agent_step()
        context_builder = setup_context_builder_step()
        run_chatbot(agent, context_builder)
        return
    
    # 데모 예측만 실행
    if args.demo:
        model_path = Path('models/iris_classifier.pkl')
        if not model_path.exists():
            print("오류: 학습된 모델이 없다. 먼저 --train으로 모델을 학습하라.")
            sys.exit(1)
        
        agent = setup_agent_step()
        context_builder = setup_context_builder_step()
        run_demo_predictions(agent, context_builder)
        return
    
    # 전체 파이프라인 실행
    model_path = Path('models/iris_classifier.pkl')
    
    # Step 1: 모델 학습 (이미 학습된 모델이 없는 경우)
    if not model_path.exists():
        train_model_step()
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
    print("Enable Agent 파이프라인 완료")
    print("=" * 60)
    
    # 최종 통계 출력
    summary = context_builder.get_summary()
    if summary:
        print(f"\n총 예측 수행: {summary['total_predictions']}회")
        print(f"생성된 파일:")
        print(f"  - models/iris_classifier.pkl")
        print(f"  - models/iris_classifier_metadata.json")
        print(f"  - skills/agent_skill.yaml")
        print(f"  - context_store/prediction_logs.json")
        print(f"  - context_store/prediction_summary.json")
        print(f"  - context_store/knowledge_base.txt")


if __name__ == '__main__':
    main()
