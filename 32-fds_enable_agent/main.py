#!/usr/bin/env python3
"""
FDS Enable Agent - 메인 실행 스크립트

이상치 탐지 기반 사기 탐지 시스템을 Enable Agent로 구현한다.

사용법:
    python main.py                  # 전체 파이프라인 실행
    python main.py --train          # 모델 학습만 실행
    python main.py --chat           # 챗봇만 실행
    python main.py --demo           # 데모 분석 실행
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def train_model_step(data_path: str):
    """모델 학습 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 1: 이상치 탐지 모델 학습")
    print("=" * 60)
    
    from src.train_model import train_model
    iso_forest, autoencoder, scaler, metadata = train_model(data_path)
    
    return metadata


def setup_agent_step():
    """Enable Agent 설정 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 2: FDS Enable Agent 설정")
    print("=" * 60)
    
    from src.enable_agent import create_skill_file, FDSEnableAgent
    
    skill_path = create_skill_file('skills/agent_skill.yaml')
    agent = FDSEnableAgent(skill_path)
    
    return agent


def setup_context_builder_step():
    """Context Builder 설정 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 3: Context Builder 설정")
    print("=" * 60)
    
    from src.context_builder import FDSContextBuilder
    
    context_builder = FDSContextBuilder('context_store')
    
    return context_builder


def run_demo_analyses(agent, context_builder):
    """데모 분석을 실행한다."""
    print("\n" + "=" * 60)
    print("Step 4: 데모 분석 실행")
    print("=" * 60)
    
    test_transactions = [
        # 정상 거래
        {
            'purchase_value': 45, 'age': 35, 'source': 'SEO', 'browser': 'Chrome', 'sex': 'M',
            'signup_time': '2024-01-15T10:30:00', 'purchase_time': '2024-02-20T14:25:00'
        },
        # 정상 거래
        {
            'purchase_value': 30, 'age': 42, 'source': 'Ads', 'browser': 'Safari', 'sex': 'F',
            'signup_time': '2024-02-01T09:00:00', 'purchase_time': '2024-02-15T11:30:00'
        },
        # 의심 거래 (빠른 구매 + 심야)
        {
            'purchase_value': 150, 'age': 22, 'source': 'Direct', 'browser': 'Opera', 'sex': 'M',
            'signup_time': '2024-03-10T02:30:00', 'purchase_time': '2024-03-10T02:45:00'
        },
        # 고위험 거래
        {
            'purchase_value': 200, 'age': 18, 'source': 'Direct', 'browser': 'IE', 'sex': 'F',
            'signup_time': '2024-03-15T03:00:00', 'purchase_time': '2024-03-15T03:01:00'
        },
    ]
    
    print("\n=== 분석 결과 ===\n")
    
    for i, tx in enumerate(test_transactions, 1):
        result = agent.analyze(tx)
        context_builder.add_analysis(result)
        
        status = "사기 의심" if result['is_fraud'] else "정상"
        print(f"[분석 {i}]")
        print(f"  금액: ${tx['purchase_value']}, 나이: {tx['age']}세, 유입: {tx['source']}")
        print(f"  결과: {status} ({result['risk_level']})")
        print(f"  점수: {result['anomaly_score']:.2%}")
        if result['risk_factors']:
            print(f"  위험 요인: {', '.join(result['risk_factors'])}")
        print()
    
    # 통계 출력
    summary = context_builder.get_summary()
    print("=== 분석 통계 ===")
    print(f"총 분석: {summary['total_analyses']}건")
    print(f"사기 탐지: {summary['fraud_detected']}건 ({summary['fraud_rate']:.1%})")
    print(f"위험도 분포: {summary['risk_distribution']}")


def run_chatbot(agent, context_builder):
    """RAG 챗봇을 실행한다."""
    print("\n" + "=" * 60)
    print("Step 5: RAG 챗봇 실행")
    print("=" * 60)
    
    from src.rag_chatbot import FDSRAGChatbot
    
    chatbot = FDSRAGChatbot(agent, context_builder)
    
    print("\n" + "=" * 60)
    print("FDS 사기 탐지 RAG 챗봇")
    print("=" * 60)
    print("종료: 'quit' 또는 'exit'")
    print("보고서: 'report'")
    print("초기화: 'reset'")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("사용자: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("챗봇을 종료한다.")
            break
        
        if user_input.lower() == 'report':
            print("\n보고서 생성 중...")
            report = chatbot.generate_report()
            print("\n" + "=" * 60)
            print("FDS 분석 보고서")
            print("=" * 60)
            print(report)
            print("=" * 60 + "\n")
            
            report_path = Path('context_store/fds_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"보고서 저장: {report_path}\n")
            continue
        
        if user_input.lower() == 'reset':
            chatbot.reset_conversation()
            continue
        
        response = chatbot.chat(user_input)
        print(f"\n챗봇: {response}\n")
    
    return chatbot


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='FDS Enable Agent - 이상치 탐지 기반 사기 탐지 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python main.py                          # 전체 파이프라인 실행
    python main.py --train                  # 모델 학습만
    python main.py --chat                   # 챗봇만 실행
    python main.py --demo                   # 데모 분석만
    python main.py --data custom_data.csv   # 커스텀 데이터 사용
        """
    )
    
    parser.add_argument('--train', action='store_true', help='모델 학습만 실행')
    parser.add_argument('--chat', action='store_true', help='챗봇만 실행')
    parser.add_argument('--demo', action='store_true', help='데모 분석만 실행')
    parser.add_argument('--data', type=str, default='Fraud_Data.csv', help='데이터 파일 경로')
    parser.add_argument('--no-chat', action='store_true', help='챗봇 실행 건너뛰기')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("FDS Enable Agent")
    print("이상치 탐지 기반 사기 탐지 시스템")
    print("=" * 60)
    
    # 모델 학습만
    if args.train:
        train_model_step(args.data)
        return
    
    # 챗봇만 (학습된 모델 필요)
    if args.chat:
        model_path = Path('models/isolation_forest.pkl')
        if not model_path.exists():
            print("오류: 학습된 모델이 없다. 먼저 --train으로 모델을 학습하라.")
            sys.exit(1)
        
        agent = setup_agent_step()
        context_builder = setup_context_builder_step()
        run_chatbot(agent, context_builder)
        return
    
    # 데모만
    if args.demo:
        model_path = Path('models/isolation_forest.pkl')
        if not model_path.exists():
            print("오류: 학습된 모델이 없다. 먼저 --train으로 모델을 학습하라.")
            sys.exit(1)
        
        agent = setup_agent_step()
        context_builder = setup_context_builder_step()
        run_demo_analyses(agent, context_builder)
        return
    
    # 전체 파이프라인
    model_path = Path('models/isolation_forest.pkl')
    
    if not model_path.exists():
        train_model_step(args.data)
    else:
        print("\n기존 학습된 모델을 사용한다.")
        print(f"모델 경로: {model_path}")
    
    agent = setup_agent_step()
    context_builder = setup_context_builder_step()
    run_demo_analyses(agent, context_builder)
    
    if not args.no_chat:
        print("\n챗봇을 시작하려면 Enter, 건너뛰려면 'skip' 입력:")
        user_choice = input().strip().lower()
        
        if user_choice != 'skip':
            run_chatbot(agent, context_builder)
    
    print("\n" + "=" * 60)
    print("FDS Enable Agent 실행 완료")
    print("=" * 60)
    
    summary = context_builder.get_summary()
    if summary:
        print(f"\n총 분석: {summary['total_analyses']}건")
        print(f"생성된 파일:")
        print(f"  - models/isolation_forest.pkl")
        print(f"  - models/autoencoder.pth")
        print(f"  - models/fds_scaler.pkl")
        print(f"  - models/label_encoders.pkl")
        print(f"  - models/fds_metadata.json")
        print(f"  - skills/agent_skill.yaml")
        print(f"  - context_store/analysis_logs.json")
        print(f"  - context_store/fds_summary.json")
        print(f"  - context_store/fds_knowledge_base.txt")


if __name__ == '__main__':
    main()
