#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iris Classification Context Builder 모듈

예측 결과를 저장하고 지식 베이스로 변환한다.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict


class PredictionContextBuilder:
    """예측 결과 Context Builder"""
    
    def __init__(self, context_dir: str = 'context_store'):
        """
        Context Builder를 초기화한다.
        
        Args:
            context_dir: 컨텍스트 저장 디렉토리
        """
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 경로
        self.logs_file = self.context_dir / 'prediction_logs.json'
        self.summary_file = self.context_dir / 'prediction_summary.json'
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
    
    def add_prediction(self, prediction_result: Dict, metadata: Dict = None):
        """
        예측 결과를 로그에 추가한다.
        
        Args:
            prediction_result: predict() 메서드의 결과
            metadata: 추가 메타데이터
        """
        log_entry = {
            'id': len(self.logs) + 1,
            'result': prediction_result,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.logs.append(log_entry)
        
        # 파일 저장
        self._save_logs()
        self._update_summary()
        self._update_knowledge_base()
        
        print(f"✓ 예측 결과 저장 (ID: {log_entry['id']}, 품종: {prediction_result.get('predicted_class', 'N/A')})")
    
    def _save_logs(self):
        """로그를 파일에 저장한다."""
        with open(self.logs_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False, default=str)
    
    def _update_summary(self):
        """요약 통계를 업데이트한다."""
        if not self.logs:
            return
        
        # 통계 계산
        total_predictions = len(self.logs)
        class_counts = defaultdict(int)
        confidence_sum = 0
        
        for log in self.logs:
            result = log['result']
            predicted_class = result.get('predicted_class', 'unknown')
            class_counts[predicted_class] += 1
            confidence_sum += result.get('confidence', 0)
        
        avg_confidence = confidence_sum / total_predictions if total_predictions > 0 else 0
        
        # 가장 많이 예측된 클래스
        most_common = max(class_counts.items(), key=lambda x: x[1]) if class_counts else ('N/A', 0)
        
        summary = {
            'total_predictions': total_predictions,
            'class_distribution': dict(class_counts),
            'average_confidence': avg_confidence,
            'most_common_class': most_common[0],
            'most_common_count': most_common[1],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _update_knowledge_base(self):
        """자연어 지식 베이스를 업데이트한다."""
        if not self.logs:
            return
        
        summary = self.get_summary()
        
        kb_content = f"""# Iris 분류 지식 베이스

## 개요
총 {summary['total_predictions']}회의 예측을 수행했다.
평균 신뢰도는 {summary['average_confidence']:.1%}이다.

## 클래스별 예측 현황
"""
        
        for class_name, count in sorted(summary['class_distribution'].items(), key=lambda x: -x[1]):
            ratio = count / summary['total_predictions'] * 100
            kb_content += f"- {class_name}: {count}회 ({ratio:.1f}%)\n"
        
        kb_content += f"""
## 가장 많이 예측된 품종
{summary['most_common_class']} ({summary['most_common_count']}회)

## 최근 예측 기록
"""
        
        for log in reversed(self.logs[-5:]):
            result = log['result']
            features = result.get('input_features', {})
            kb_content += f"- #{log['id']}: {result['predicted_class']} "
            kb_content += f"(신뢰도: {result.get('confidence', 0):.1%}) "
            kb_content += f"- 꽃잎: {features.get('petal length (cm)', 0):.1f}x{features.get('petal width (cm)', 0):.1f}cm\n"
        
        kb_content += f"""
## Iris 품종 정보

### Setosa (부채붓꽃)
- 꽃잎이 작고 꽃받침이 넓은 품종이다.
- 특성: 꽃잎 길이 < 2cm, 꽃잎 너비 < 0.5cm
- 다른 품종과 명확하게 구분된다.

### Versicolor (버시컬러 붓꽃)
- 중간 크기의 꽃잎을 가진 품종이다.
- 특성: 꽃잎 길이 3-5cm, 꽃잎 너비 1-1.5cm
- Virginica와 일부 겹치는 특성이 있다.

### Virginica (버지니카 붓꽃)
- 꽃잎이 크고 길쭉한 품종이다.
- 특성: 꽃잎 길이 > 5cm, 꽃잎 너비 > 1.5cm
- 가장 큰 꽃잎을 가진다.

## 특성 중요도
모델에서 가장 중요한 특성은 꽃잎 너비(petal width)와 꽃잎 길이(petal length)이다.
꽃받침 특성은 상대적으로 덜 중요하다.

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
            'total_predictions': 0,
            'class_distribution': {},
            'average_confidence': 0,
            'most_common_class': 'N/A',
            'most_common_count': 0
        }
    
    def get_knowledge_base_content(self) -> str:
        """지식 베이스 내용을 반환한다."""
        if self.knowledge_base_file.exists():
            return self.knowledge_base_file.read_text(encoding='utf-8')
        return ""
    
    def get_recent_predictions(self, n: int = 5) -> List[Dict]:
        """최근 예측 결과를 반환한다."""
        return self.logs[-n:] if self.logs else []
    
    def get_class_statistics(self) -> Dict[str, Any]:
        """클래스별 상세 통계를 반환한다."""
        class_stats = defaultdict(lambda: {
            'count': 0,
            'total_confidence': 0,
            'avg_features': {
                'sepal_length': 0,
                'sepal_width': 0,
                'petal_length': 0,
                'petal_width': 0
            }
        })
        
        for log in self.logs:
            result = log['result']
            class_name = result.get('predicted_class', 'unknown')
            stats = class_stats[class_name]
            
            stats['count'] += 1
            stats['total_confidence'] += result.get('confidence', 0)
            
            features = result.get('input_features', {})
            for key in ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']:
                short_key = key.split(' ')[0] + '_' + key.split(' ')[1]
                stats['avg_features'][short_key] += features.get(key, 0)
        
        # 평균 계산
        for class_name, stats in class_stats.items():
            if stats['count'] > 0:
                stats['avg_confidence'] = stats['total_confidence'] / stats['count']
                for key in stats['avg_features']:
                    stats['avg_features'][key] /= stats['count']
        
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
    builder = PredictionContextBuilder()
    
    # 테스트 데이터 추가
    test_result = {
        'predicted_class': 'setosa',
        'korean_name': '부채붓꽃',
        'confidence': 0.98,
        'probabilities': {'setosa': 0.98, 'versicolor': 0.01, 'virginica': 0.01},
        'input_features': {
            'sepal length (cm)': 5.1,
            'sepal width (cm)': 3.5,
            'petal length (cm)': 1.4,
            'petal width (cm)': 0.2
        },
        'timestamp': datetime.now().isoformat()
    }
    
    builder.add_prediction(test_result)
    
    print("\n=== 요약 ===")
    print(json.dumps(builder.get_summary(), indent=2, ensure_ascii=False))
