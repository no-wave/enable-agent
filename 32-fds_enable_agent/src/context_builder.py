"""
FDS Context Builder 모듈

분석 결과를 저장하고 지식 베이스로 변환한다.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class FDSContextBuilder:
    """FDS 분석 결과를 컨텍스트로 저장하고 관리하는 클래스"""
    
    def __init__(self, context_dir: str = 'context_store'):
        """Context Builder를 초기화한다."""
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(exist_ok=True)
        
        self.log_file = self.context_dir / 'analysis_logs.json'
        self.summary_file = self.context_dir / 'fds_summary.json'
        self.knowledge_base_file = self.context_dir / 'fds_knowledge_base.txt'
        
        self.logs = self._load_logs()
        
        print(f"FDS Context Builder 초기화 완료")
        print(f"저장된 분석 로그: {len(self.logs)}개")
    
    def _load_logs(self) -> List[Dict[str, Any]]:
        """기존 로그를 로드한다."""
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def add_analysis(self, result: Dict[str, Any]):
        """새로운 분석 결과를 저장한다."""
        self.logs.append(result)
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"분석 결과 저장 완료 (총 {len(self.logs)}개)")
        
        self._update_summary()
        self._update_knowledge_base()
    
    def _update_summary(self):
        """분석 결과 요약을 업데이트한다."""
        if not self.logs:
            return
        
        total = len(self.logs)
        fraud_count = sum(1 for l in self.logs if l['is_fraud'])
        
        risk_dist = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for log in self.logs:
            risk_dist[log['risk_level']] += 1
        
        avg_score = sum(l['anomaly_score'] for l in self.logs) / total
        avg_amount = sum(l['input_data']['purchase_value'] for l in self.logs) / total
        
        fraud_logs = [l for l in self.logs if l['is_fraud']]
        normal_logs = [l for l in self.logs if not l['is_fraud']]
        
        fraud_avg_time = sum(l['derived_features']['time_diff_hours'] for l in fraud_logs) / len(fraud_logs) if fraud_logs else 0
        normal_avg_time = sum(l['derived_features']['time_diff_hours'] for l in normal_logs) / len(normal_logs) if normal_logs else 0
        
        summary = {
            "total_analyses": total,
            "fraud_detected": fraud_count,
            "fraud_rate": fraud_count / total,
            "risk_distribution": risk_dist,
            "average_anomaly_score": avg_score,
            "average_purchase_value": avg_amount,
            "fraud_avg_time_diff_hours": fraud_avg_time,
            "normal_avg_time_diff_hours": normal_avg_time,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _update_knowledge_base(self):
        """자연어 지식 베이스를 업데이트한다."""
        if not self.logs:
            return
        
        with open(self.summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        knowledge = f"""# FDS 사기 탐지 지식 베이스

## 전체 통계 (마지막 업데이트: {summary['last_updated']})

- 총 분석 건수: {summary['total_analyses']}건
- 사기 탐지 건수: {summary['fraud_detected']}건
- 사기 탐지율: {summary['fraud_rate']:.2%}
- 평균 이상치 점수: {summary['average_anomaly_score']:.2%}
- 평균 거래 금액: ${summary['average_purchase_value']:.1f}

## 위험도 분포

- LOW: {summary['risk_distribution']['LOW']}건
- MEDIUM: {summary['risk_distribution']['MEDIUM']}건
- HIGH: {summary['risk_distribution']['HIGH']}건
- CRITICAL: {summary['risk_distribution']['CRITICAL']}건

## 패턴 분석

- 사기 거래 평균 가입-구매 시간차: {summary['fraud_avg_time_diff_hours']:.1f}시간
- 정상 거래 평균 가입-구매 시간차: {summary['normal_avg_time_diff_hours']:.1f}시간

## 최근 분석 기록 (최근 10건)

"""
        
        for i, log in enumerate(reversed(self.logs[-10:]), 1):
            status = "사기 의심" if log['is_fraud'] else "정상"
            factors = ', '.join(log['risk_factors']) if log['risk_factors'] else '없음'
            knowledge += f"""### 분석 #{len(self.logs) - i + 1}
- 결과: {status} ({log['risk_level']})
- 점수: {log['anomaly_score']:.2%}
- 금액: ${log['input_data']['purchase_value']}
- 시간차: {log['derived_features']['time_diff_hours']:.1f}시간
- 위험 요인: {factors}

"""
        
        with open(self.knowledge_base_file, 'w', encoding='utf-8') as f:
            f.write(knowledge.strip())
    
    def get_knowledge_base_content(self) -> str:
        """지식 베이스 내용을 반환한다."""
        if self.knowledge_base_file.exists():
            with open(self.knowledge_base_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def get_summary(self) -> Dict[str, Any]:
        """요약 정보를 반환한다."""
        if self.summary_file.exists():
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """전체 로그를 반환한다."""
        return self.logs
    
    def clear_logs(self):
        """로그를 초기화한다."""
        self.logs = []
        for f in [self.log_file, self.summary_file, self.knowledge_base_file]:
            if f.exists():
                f.unlink()
        print("로그 초기화 완료")
    
    def get_statistics(self) -> Dict[str, Any]:
        """상세 통계를 반환한다."""
        if not self.logs:
            return {}
        
        fraud_logs = [l for l in self.logs if l['is_fraud']]
        normal_logs = [l for l in self.logs if not l['is_fraud']]
        
        def avg_value(logs, key):
            if not logs:
                return 0
            return sum(l['input_data'][key] for l in logs) / len(logs)
        
        def night_rate(logs):
            if not logs:
                return 0
            return sum(1 for l in logs if l['derived_features']['is_night']) / len(logs)
        
        return {
            'total_analyses': len(self.logs),
            'fraud': {
                'count': len(fraud_logs),
                'avg_amount': avg_value(fraud_logs, 'purchase_value'),
                'night_rate': night_rate(fraud_logs)
            },
            'normal': {
                'count': len(normal_logs),
                'avg_amount': avg_value(normal_logs, 'purchase_value'),
                'night_rate': night_rate(normal_logs)
            }
        }


if __name__ == '__main__':
    builder = FDSContextBuilder()
    
    # 테스트 데이터 추가
    test_result = {
        "is_fraud": True,
        "risk_level": "HIGH",
        "anomaly_score": 0.75,
        "risk_factors": ["매우 빠른 구매", "심야 거래"],
        "input_data": {"purchase_value": 150, "age": 22, "source": "Direct", "browser": "Opera", "sex": "M"},
        "derived_features": {"time_diff_hours": 0.5, "is_night": 1},
        "timestamp": datetime.now().isoformat()
    }
    
    builder.add_analysis(test_result)
    print("\n=== 지식 베이스 ===")
    print(builder.get_knowledge_base_content())
