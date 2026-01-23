#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Plastic Detection Enable Agent - 메인 실행 스크립트

이 스크립트는 전체 Enable Agent 파이프라인을 실행한다:
1. 데이터 준비 및 YOLO 모델 학습
2. Enable Agent 초기화
3. Context Builder 설정
4. RAG 챗봇 실행

사용법:
    python main.py                  # 전체 파이프라인 실행
    python main.py --train          # 모델 학습만 실행
    python main.py --chat           # 챗봇만 실행 (학습된 모델 필요)
    python main.py --demo           # 데모 탐지 실행
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


def train_model_step(raw_data_dir: str = 'data/raw_data'):
    """모델 학습 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 1: 데이터 준비 및 YOLOv8 모델 학습")
    print("=" * 60)
    
    from src.train_model import prepare_yolo_dataset, train_model
    
    # 데이터셋 준비
    dataset_info = prepare_yolo_dataset(raw_data_dir=raw_data_dir)
    
    if dataset_info is None:
        print("\n⚠️  데이터셋 준비 실패")
        print("data/raw_data 디렉토리에 다음 구조로 데이터를 넣어달라:")
        print("  raw_data/")
        print("  ├── annotations/  # COCO JSON 파일들")
        print("  └── images/       # 이미지 파일들")
        return None
    
    # 모델 학습
    model, results, metadata = train_model(
        data_yaml=dataset_info['yaml_path'],
        epochs=50
    )
    
    return model, results, metadata


def setup_agent_step():
    """Enable Agent 설정 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 2: Enable Agent 설정")
    print("=" * 60)
    
    from src.enable_agent import PlasticDetectionAgent, create_skill_file
    
    # 스킬 파일 생성
    skill_path = create_skill_file('skills/agent_skill.yaml')
    
    # Agent 초기화
    agent = PlasticDetectionAgent(skill_path)
    
    return agent


def setup_context_builder_step():
    """Context Builder 설정 단계를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 3: Context Builder 설정")
    print("=" * 60)
    
    from src.context_builder import DetectionContextBuilder
    
    context_builder = DetectionContextBuilder('context_store')
    
    return context_builder


def run_demo_detection(agent, context_builder):
    """데모 탐지를 실행한다."""
    print("\n" + "=" * 60)
    print("Step 4: 데모 탐지 실행")
    print("=" * 60)
    
    # 검증 이미지 찾기
    val_images = list(Path('datasets/plastic/val/images').glob('*.jpg'))
    
    if not val_images:
        print("⚠️  검증 이미지가 없다.")
        return
    
    print(f"\n{len(val_images)}개의 검증 이미지 중 최대 5개 분석...\n")
    
    for i, img_path in enumerate(val_images[:5], 1):
        print(f"[탐지 {i}] {img_path.name}")
        
        result = agent.detect(str(img_path))
        context_builder.add_detection(result, metadata={'demo': True, 'index': i})
        
        if result['detections']:
            print(f"  탐지: {result['num_detections']}개")
            print(f"  클래스: {result['class_counts']}")
            print(f"  재활용 가능: {result['recyclable_count']}개")
        else:
            print("  탐지된 객체 없음")
        print()
    
    # 통계 출력
    summary = context_builder.get_summary()
    print("=== 탐지 통계 ===")
    print(f"총 분석 이미지: {summary['total_images']}개")
    print(f"총 탐지 객체: {summary['total_detections']}개")
    print(f"재활용 가능: {summary['total_recyclable']}개")
    print(f"재활용 불가: {summary['total_non_recyclable']}개")
    print(f"클래스 분포: {summary['class_distribution']}")


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
        description='YOLO Plastic Detection Enable Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python main.py                          # 전체 파이프라인 실행
    python main.py --train                  # 모델 학습만
    python main.py --chat                   # 챗봇만 실행
    python main.py --demo                   # 데모 탐지만
    python main.py --data data/raw_data     # 커스텀 데이터 경로
        """
    )
    
    parser.add_argument('--train', action='store_true', help='모델 학습만 실행')
    parser.add_argument('--chat', action='store_true', help='챗봇만 실행')
    parser.add_argument('--demo', action='store_true', help='데모 탐지만 실행')
    parser.add_argument('--data', type=str, default='data/raw_data', help='원본 데이터 경로')
    parser.add_argument('--no-chat', action='store_true', help='챗봇 실행 건너뛰기')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("YOLO Plastic Detection Enable Agent")
    print("플라스틱 재활용 분류를 위한 객체 탐지 시스템")
    print("=" * 60)
    
    # 모델 학습만 실행
    if args.train:
        train_model_step(args.data)
        return
    
    # 챗봇만 실행 (학습된 모델 필요)
    if args.chat:
        model_path = Path('models/plastic_yolov8_best.pt')
        if not model_path.exists():
            print("오류: 학습된 모델이 없다. 먼저 --train으로 모델을 학습하라.")
            sys.exit(1)
        
        agent = setup_agent_step()
        context_builder = setup_context_builder_step()
        run_chatbot(agent, context_builder)
        return
    
    # 데모 탐지만 실행
    if args.demo:
        model_path = Path('models/plastic_yolov8_best.pt')
        if not model_path.exists():
            print("오류: 학습된 모델이 없다. 먼저 --train으로 모델을 학습하라.")
            sys.exit(1)
        
        agent = setup_agent_step()
        context_builder = setup_context_builder_step()
        run_demo_detection(agent, context_builder)
        return
    
    # 전체 파이프라인 실행
    model_path = Path('models/plastic_yolov8_best.pt')
    
    # Step 1: 모델 학습 (이미 학습된 모델이 없는 경우)
    if not model_path.exists():
        result = train_model_step(args.data)
        if result is None:
            print("\n학습 실패. 데이터를 확인해달라.")
            return
    else:
        print("\n기존 학습된 모델을 사용한다.")
        print(f"모델 경로: {model_path}")
    
    # Step 2: Enable Agent 설정
    agent = setup_agent_step()
    
    # Step 3: Context Builder 설정
    context_builder = setup_context_builder_step()
    
    # Step 4: 데모 탐지
    run_demo_detection(agent, context_builder)
    
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
        print(f"\n총 탐지 수행: {summary['total_detections']}회")
        print(f"생성된 파일:")
        print(f"  - models/plastic_yolov8_best.pt")
        print(f"  - models/yolo_metadata.json")
        print(f"  - skills/agent_skill.yaml")
        print(f"  - context_store/detection_logs.json")
        print(f"  - context_store/detection_summary.json")
        print(f"  - context_store/knowledge_base.txt")


if __name__ == '__main__':
    main()
