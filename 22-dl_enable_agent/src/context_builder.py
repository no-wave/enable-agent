"""
Context Builder 모듈

예측 결과를 저장하고 지식 베이스로 변환한다.
RAG를 위한 컨텍스트를 관리한다.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class PredictionContextBuilder:
    """예측 결과를 컨텍스트로 저장하고 관리하는 클래스"""
    
    def __init__(self, context_dir: str = 'context_store'):
        """Context Builder를 초기화한다."""
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(exist_ok=True)
        
        self.log_file = self.context_dir / 'prediction_logs.json'
        self.summary_file = self.context_dir / 'prediction_summary.json'
        self.knowledge_base_file = self.context_dir / 'knowledge_base.txt'
        
        self.logs = self._load_logs()
        
        print(f"Context Builder 초기화 완료")
        print(f"저장된 예측 로그: {len(self.logs)}개")
    
    def _load_logs(self) -> List[Dict[str, Any]]:
        """기존 로그를 로드한다."""
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def add_prediction(self, prediction_result: Dict[str, Any]):
        """새로운 예측 결과를 컨텍스트에 추가한다."""
        self.logs.append(prediction_result)
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        
        print(f"예측 결과 저장 완료 (총 {len(self.logs)}개)")
        
        self._update_summary()
        self._update_knowledge_base()
    
    def _update_summary(self):
        """예측 결과 요약을 업데이트한다."""
        if not self.logs:
            return
        
        total_predictions = len(self.logs)
        
        class_counts = {'Transported': 0, 'Not Transported': 0}
        probability_sum = 0
        confidence_sum = 0
        
        age_values = []
        spending_totals = []
        homeplanet_counts = {}
        cryosleep_counts = {0: 0, 1: 0}
        
        for log in self.logs:
            predicted_class = log['predicted_class']
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            probability_sum += log['probability']
            confidence_sum += log['confidence']
            
            features = log['input_features']
            age_values.append(features['Age'])
            
            total_spending = (features['RoomService'] + features['FoodCourt'] +
                              features['ShoppingMall'] + features['Spa'] + features['VRDeck'])
            spending_totals.append(total_spending)
            
            hp = features['HomePlanet']
            homeplanet_counts[hp] = homeplanet_counts.get(hp, 0) + 1
            cryosleep_counts[features['CryoSleep']] += 1
        
        avg_probability = probability_sum / total_predictions
        avg_confidence = confidence_sum / total_predictions
        avg_age = sum(age_values) / len(age_values)
        avg_spending = sum(spending_totals) / len(spending_totals)
        
        summary = {
            "total_predictions": total_predictions,
            "class_distribution": class_counts,
            "average_probability": avg_probability,
            "average_confidence": avg_confidence,
            "average_age": avg_age,
            "average_total_spending": avg_spending,
            "homeplanet_distribution": homeplanet_counts,
            "cryosleep_distribution": cryosleep_counts,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _update_knowledge_base(self):
        """예측 결과를 자연어 지식 베이스로 변환한다."""
        if not self.logs:
            return
        
        with open(self.summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        homeplanet_map = {0: 'Earth', 1: 'Europa', 2: 'Mars', 3: 'Unknown'}
        
        knowledge_text = f"""# Spaceship Titanic 예측 지식 베이스

## 전체 통계 (마지막 업데이트: {summary['last_updated']})

총 예측 수행 횟수: {summary['total_predictions']}회
평균 이동 확률: {summary['average_probability']:.2%}
평균 예측 신뢰도: {summary['average_confidence']:.2%}
평균 나이: {summary['average_age']:.1f}세
평균 총 지출: {summary['average_total_spending']:.1f}

## 예측 결과 분포

"""
        
        for cls, count in summary['class_distribution'].items():
            percentage = (count / summary['total_predictions']) * 100
            knowledge_text += f"- {cls}: {count}회 ({percentage:.1f}%)\n"
        
        knowledge_text += "\n## 출발 행성 분포\n\n"
        for hp, count in summary['homeplanet_distribution'].items():
            planet_name = homeplanet_map.get(int(hp), 'Unknown')
            knowledge_text += f"- {planet_name}: {count}회\n"
        
        knowledge_text += "\n## 냉동 수면 분포\n\n"
        knowledge_text += f"- 냉동 수면 O: {summary['cryosleep_distribution'].get(1, 0)}회\n"
        knowledge_text += f"- 냉동 수면 X: {summary['cryosleep_distribution'].get(0, 0)}회\n"
        
        knowledge_text += "\n## 최근 예측 기록 (최근 10개)\n\n"
        
        recent_logs = self.logs[-10:]
        for i, log in enumerate(reversed(recent_logs), 1):
            decoded = log.get('decoded_features', log['input_features'])
            knowledge_text += f"""### 예측 #{len(self.logs) - i + 1}

- 예측 결과: {log['predicted_class']}
- 이동 확률: {log['probability']:.2%}
- 신뢰도: {log['confidence']:.2%}
- 출발 행성: {decoded.get('HomePlanet', log['input_features']['HomePlanet'])}
- 냉동 수면: {decoded.get('CryoSleep', log['input_features']['CryoSleep'])}
- 나이: {log['input_features']['Age']}세

"""
        
        with open(self.knowledge_base_file, 'w', encoding='utf-8') as f:
            f.write(knowledge_text.strip())
    
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
        for file in [self.log_file, self.summary_file, self.knowledge_base_file]:
            if file.exists():
                file.unlink()
        print("로그 초기화 완료")
    
    def get_statistics(self) -> Dict[str, Any]:
        """상세 통계를 반환한다."""
        if not self.logs:
            return {}
        
        transported_logs = [l for l in self.logs if l['predicted_class'] == 'Transported']
        not_transported_logs = [l for l in self.logs if l['predicted_class'] == 'Not Transported']
        
        def avg_spending(logs):
            if not logs:
                return 0
            total = sum(
                l['input_features']['RoomService'] + l['input_features']['FoodCourt'] +
                l['input_features']['ShoppingMall'] + l['input_features']['Spa'] + l['input_features']['VRDeck']
                for l in logs
            )
            return total / len(logs)
        
        def avg_age(logs):
            if not logs:
                return 0
            return sum(l['input_features']['Age'] for l in logs) / len(logs)
        
        def cryosleep_rate(logs):
            if not logs:
                return 0
            return sum(1 for l in logs if l['input_features']['CryoSleep'] == 1) / len(logs)
        
        return {
            'total_predictions': len(self.logs),
            'transported': {
                'count': len(transported_logs),
                'avg_age': avg_age(transported_logs),
                'avg_spending': avg_spending(transported_logs),
                'cryosleep_rate': cryosleep_rate(transported_logs)
            },
            'not_transported': {
                'count': len(not_transported_logs),
                'avg_age': avg_age(not_transported_logs),
                'avg_spending': avg_spending(not_transported_logs),
                'cryosleep_rate': cryosleep_rate(not_transported_logs)
            }
        }


if __name__ == '__main__':
    builder = PredictionContextBuilder()
    
    # 테스트 예측 추가
    test_predictions = [
        {
            "predicted_class": "Transported",
            "probability": 0.85,
            "confidence": 0.85,
            "input_features": {
                "HomePlanet": 0, "CryoSleep": 1, "Destination": 2,
                "Age": 25.0, "VIP": 0, "RoomService": 0.0, "FoodCourt": 0.0,
                "ShoppingMall": 0.0, "Spa": 0.0, "VRDeck": 0.0
            },
            "decoded_features": {
                "HomePlanet": "Earth", "CryoSleep": "Yes", "Destination": "55 Cancri e",
                "Age": 25.0, "VIP": "No", "RoomService": 0.0, "FoodCourt": 0.0,
                "ShoppingMall": 0.0, "Spa": 0.0, "VRDeck": 0.0
            },
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    for pred in test_predictions:
        builder.add_prediction(pred)
    
    print("\n=== 지식 베이스 ===")
    print(builder.get_knowledge_base_content()[:1000])
