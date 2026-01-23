#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Plastic Detection Context Builder 모듈

탐지 결과를 저장하고 지식 베이스로 변환한다.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict


class DetectionContextBuilder:
    """플라스틱 탐지 결과 Context Builder"""
    
    def __init__(self, context_dir: str = 'context_store'):
        """
        Context Builder를 초기화한다.
        
        Args:
            context_dir: 컨텍스트 저장 디렉토리
        """
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 경로
        self.logs_file = self.context_dir / 'detection_logs.json'
        self.summary_file = self.context_dir / 'detection_summary.json'
        self.knowledge_base_file = self.context_dir / 'knowledge_base.txt'
        
        # 로그 로드
        self.logs = self._load_logs()
        
        print(f"✓ Context Builder 초기화 완료")
        print(f"  저장 경로: {self.context_dir}")
        print(f"  기존 로그: {len(self.logs)}개")
    
    def _load_logs(self) -> List[Dict]:
        """저장된 로그를 로드한다."""
        if self.logs_file.exists():
            with open(self.logs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def add_detection(self, detection_result: Dict, metadata: Dict = None):
        """
        탐지 결과를 로그에 추가한다.
        
        Args:
            detection_result: detect() 메서드의 결과
            metadata: 추가 메타데이터
        """
        log_entry = {
            'id': len(self.logs) + 1,
            'result': detection_result,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.logs.append(log_entry)
        
        # 파일 저장
        self._save_logs()
        self._update_summary()
        self._update_knowledge_base()
        
        print(f"✓ 탐지 결과 저장 (ID: {log_entry['id']}, 객체: {detection_result.get('num_detections', 0)}개)")
    
    def _save_logs(self):
        """로그를 파일에 저장한다."""
        with open(self.logs_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False, default=str)
    
    def _update_summary(self):
        """요약 통계를 업데이트한다."""
        if not self.logs:
            return
        
        # 통계 계산
        total_detections = 0
        total_recyclable = 0
        total_non_recyclable = 0
        class_counts = defaultdict(int)
        confidence_sum = 0
        detection_count = 0
        
        for log in self.logs:
            result = log['result']
            total_detections += result.get('num_detections', 0)
            total_recyclable += result.get('recyclable_count', 0)
            total_non_recyclable += result.get('non_recyclable_count', 0)
            
            # 클래스별 통계
            for class_name, count in result.get('class_counts', {}).items():
                class_counts[class_name] += count
            
            # 평균 신뢰도
            for det in result.get('detections', []):
                confidence_sum += det.get('confidence', 0)
                detection_count += 1
        
        avg_confidence = confidence_sum / detection_count if detection_count > 0 else 0
        
        summary = {
            'total_images': len(self.logs),
            'total_detections': total_detections,
            'total_recyclable': total_recyclable,
            'total_non_recyclable': total_non_recyclable,
            'recyclable_ratio': total_recyclable / total_detections if total_detections > 0 else 0,
            'class_distribution': dict(class_counts),
            'average_confidence': avg_confidence,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _update_knowledge_base(self):
        """자연어 지식 베이스를 업데이트한다."""
        if not self.logs:
            return
        
        summary = self.get_summary()
        
        kb_content = f"""# 플라스틱 탐지 지식 베이스

## 개요
총 {summary['total_images']}개의 이미지를 분석하여 {summary['total_detections']}개의 플라스틱 객체를 탐지했다.

## 통계
- 재활용 가능: {summary['total_recyclable']}개 ({summary['recyclable_ratio']:.1%})
- 재활용 불가: {summary['total_non_recyclable']}개 ({1 - summary['recyclable_ratio']:.1%})
- 평균 탐지 신뢰도: {summary['average_confidence']:.1%}

## 클래스별 탐지 현황
"""
        
        for class_name, count in sorted(summary['class_distribution'].items(), key=lambda x: -x[1]):
            kb_content += f"- {class_name}: {count}개\n"
        
        kb_content += f"""
## 최근 탐지 기록
"""
        
        for log in reversed(self.logs[-5:]):
            result = log['result']
            kb_content += f"- #{log['id']}: {result.get('num_detections', 0)}개 탐지 "
            if result.get('class_counts'):
                classes = ', '.join([f"{k}:{v}" for k, v in result['class_counts'].items()])
                kb_content += f"({classes})"
            kb_content += f" - {log['timestamp']}\n"
        
        kb_content += f"""
## 재활용 가이드

### PET (폴리에틸렌 테레프탈레이트) - #1
- 재활용 가능
- 예시: 음료수 병, 생수 병
- 배출: 라벨을 제거하고 내용물을 비운 후 압축하여 배출한다.

### PS (폴리스타이렌) - #6
- 재활용 불가 (일반 쓰레기)
- 예시: 요거트 컵, 스티로폼
- 배출: 일반 쓰레기로 배출한다. 스티로폼은 별도 배출한다.

### PP (폴리프로필렌) - #5
- 재활용 가능
- 예시: 식품 용기, 병뚜껑
- 배출: 깨끗이 세척 후 플라스틱류로 배출한다.

### PE (폴리에틸렌) - #2, #4
- 재활용 가능
- 예시: 비닐봉지, 세제 용기
- 배출: HDPE는 플라스틱류, LDPE(비닐)는 비닐류로 배출한다.

마지막 업데이트: {summary['last_updated']}
"""
        
        with open(self.knowledge_base_file, 'w', encoding='utf-8') as f:
            f.write(kb_content)
    
    def get_summary(self) -> Dict[str, Any]:
        """요약 통계를 반환한다."""
        if self.summary_file.exists():
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'total_images': 0,
            'total_detections': 0,
            'total_recyclable': 0,
            'total_non_recyclable': 0,
            'recyclable_ratio': 0,
            'class_distribution': {},
            'average_confidence': 0
        }
    
    def get_knowledge_base_content(self) -> str:
        """지식 베이스 내용을 반환한다."""
        if self.knowledge_base_file.exists():
            return self.knowledge_base_file.read_text(encoding='utf-8')
        return ""
    
    def get_recent_detections(self, n: int = 5) -> List[Dict]:
        """최근 탐지 결과를 반환한다."""
        return self.logs[-n:] if self.logs else []
    
    def get_class_statistics(self) -> Dict[str, Any]:
        """클래스별 상세 통계를 반환한다."""
        class_stats = defaultdict(lambda: {
            'count': 0,
            'total_confidence': 0,
            'recyclable': None
        })
        
        for log in self.logs:
            for det in log['result'].get('detections', []):
                class_name = det['class_name']
                class_stats[class_name]['count'] += 1
                class_stats[class_name]['total_confidence'] += det.get('confidence', 0)
                class_stats[class_name]['recyclable'] = det.get('recyclable', False)
        
        # 평균 계산
        for class_name, stats in class_stats.items():
            if stats['count'] > 0:
                stats['avg_confidence'] = stats['total_confidence'] / stats['count']
        
        return dict(class_stats)
    
    def clear_logs(self):
        """모든 로그를 초기화한다."""
        self.logs = []
        
        for file_path in [self.logs_file, self.summary_file, self.knowledge_base_file]:
            if file_path.exists():
                file_path.unlink()
        
        print("✓ 로그 초기화 완료")


if __name__ == '__main__':
    # Context Builder 테스트
    builder = DetectionContextBuilder()
    
    # 테스트 데이터 추가
    test_result = {
        'image_path': 'test.jpg',
        'detections': [
            {'class_name': 'PET', 'confidence': 0.95, 'recyclable': True},
            {'class_name': 'PS', 'confidence': 0.87, 'recyclable': False}
        ],
        'num_detections': 2,
        'class_counts': {'PET': 1, 'PS': 1},
        'recyclable_count': 1,
        'non_recyclable_count': 1,
        'timestamp': datetime.now().isoformat()
    }
    
    builder.add_detection(test_result)
    
    print("\n=== 요약 ===")
    print(json.dumps(builder.get_summary(), indent=2, ensure_ascii=False))
    
    print("\n=== 지식 베이스 ===")
    print(builder.get_knowledge_base_content()[:500] + "...")
