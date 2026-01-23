"""
FDS Enable Agent 구현 모듈

이상치 탐지 모델을 Enable Agent로 래핑하여 LLM 기반 시스템에 통합한다.
"""

import json
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn

from openai import OpenAI


class FraudAutoencoder(nn.Module):
    """이상치 탐지용 Autoencoder"""
    
    def __init__(self, input_dim: int = 10, latent_dim: int = 4):
        super(FraudAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            return torch.mean((x - self.forward(x)) ** 2, dim=1)


class FDSEnableAgent:
    """이상치 탐지 기반 사기 탐지 Enable Agent"""
    
    def __init__(self, skill_path: str):
        """Enable Agent를 초기화한다."""
        with open(skill_path, 'r', encoding='utf-8') as f:
            self.skill = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Isolation Forest 로드
        self.iso_forest = joblib.load(self.skill['model_info']['isolation_forest_path'])
        
        # Autoencoder 로드
        checkpoint = torch.load(
            self.skill['model_info']['autoencoder_path'],
            map_location=self.device,
            weights_only=True
        )
        self.autoencoder = FraudAutoencoder(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim']
        ).to(self.device)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.autoencoder.eval()
        self.ae_threshold = checkpoint['threshold']
        
        # 스케일러 및 인코더 로드
        self.scaler = joblib.load(self.skill['model_info']['scaler_path'])
        self.label_encoders = joblib.load(self.skill['model_info']['label_encoders_path'])
        
        # 메타데이터 로드
        with open(self.skill['model_info']['metadata_path'], 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.risk_thresholds = self.skill['risk_thresholds']
        self.client = OpenAI()
        
        print(f"FDS Enable Agent 초기화 완료: {self.skill['agent_name']}")
        print(f"앙상블 임계값: {self.metadata['ensemble_threshold']:.4f}")
        print(f"디바이스: {self.device}")
    
    def get_capability_description(self) -> str:
        """에이전트의 기능을 자연어로 반환한다."""
        return f"""
Agent: {self.skill['agent_name']} v{self.skill['version']}
설명: {self.skill['description']}

주요 기능: {self.skill['capabilities']['primary']}
부가 기능: {', '.join(self.skill['capabilities']['secondary'])}

입력 파라미터:
- purchase_value: 구매 금액
- age: 사용자 나이
- source: 유입 경로 (SEO/Ads/Direct)
- browser: 브라우저 (Chrome/Safari/FireFox/IE/Opera)
- sex: 성별 (M/F)
- signup_time: 가입 시간 (ISO format)
- purchase_time: 구매 시간 (ISO format)

출력:
- is_fraud: 사기 여부
- risk_level: 위험도 등급 (LOW/MEDIUM/HIGH/CRITICAL)
- anomaly_score: 이상치 점수 (0-1)
- risk_factors: 위험 요인 목록

모델 AUC-ROC: {self.metadata['test_auc_roc']:.4f}
""".strip()
    
    def _engineer_features(self, input_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """입력 데이터로부터 특성을 엔지니어링한다."""
        signup_time = pd.to_datetime(input_data['signup_time'])
        purchase_time = pd.to_datetime(input_data['purchase_time'])
        
        derived = {
            'time_diff_hours': (purchase_time - signup_time).total_seconds() / 3600,
            'signup_hour': signup_time.hour,
            'purchase_hour': purchase_time.hour,
            'is_weekend': 1 if purchase_time.dayofweek in [5, 6] else 0,
            'is_night': 1 if 0 <= purchase_time.hour < 6 else 0
        }
        
        features = np.array([[
            input_data['purchase_value'],
            input_data['age'],
            derived['time_diff_hours'],
            derived['signup_hour'],
            derived['purchase_hour'],
            derived['is_weekend'],
            derived['is_night'],
            self.label_encoders['source'].transform([input_data['source']])[0],
            self.label_encoders['browser'].transform([input_data['browser']])[0],
            self.label_encoders['sex'].transform([input_data['sex']])[0]
        ]])
        
        return features, derived
    
    def _calculate_risk_factors(self, input_data: Dict, derived: Dict, score: float) -> List[str]:
        """위험 요인을 분석한다."""
        factors = []
        
        if derived['time_diff_hours'] < 1:
            factors.append("매우 빠른 구매 (가입 후 1시간 이내)")
        elif derived['time_diff_hours'] < 24:
            factors.append("빠른 구매 (가입 후 24시간 이내)")
        
        if derived['is_night']:
            factors.append("심야 시간대 거래 (00:00-06:00)")
        
        if input_data['purchase_value'] > 100:
            factors.append(f"고액 거래 (${input_data['purchase_value']})")
        
        if input_data['age'] < 20 or input_data['age'] > 60:
            factors.append(f"비일반적 연령대 ({input_data['age']}세)")
        
        if input_data['source'] == 'Direct':
            factors.append("직접 유입 (Direct)")
        
        if not factors and score > 0.3:
            factors.append("모델이 탐지한 이상 패턴")
        
        return factors
    
    def _get_risk_level(self, score: float) -> str:
        """이상치 점수를 기반으로 위험도 등급을 반환한다."""
        if score >= self.risk_thresholds['critical']:
            return 'CRITICAL'
        elif score >= self.risk_thresholds['high']:
            return 'HIGH'
        elif score >= self.risk_thresholds['medium']:
            return 'MEDIUM'
        return 'LOW'
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """거래를 분석하고 사기 가능성을 평가한다."""
        features, derived = self._engineer_features(input_data)
        features_scaled = self.scaler.transform(features)
        
        # Isolation Forest 점수
        iso_score = -self.iso_forest.score_samples(features_scaled)[0]
        
        # Autoencoder 점수
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        ae_score = self.autoencoder.get_reconstruction_error(features_tensor).cpu().numpy()[0]
        
        # 점수 정규화 및 앙상블
        iso_norm = 1 / (1 + np.exp(-2 * (iso_score - 0.5)))
        ae_norm = 1 / (1 + np.exp(-2 * (ae_score / self.ae_threshold - 1)))
        anomaly_score = (iso_norm + ae_norm) / 2
        
        is_fraud = anomaly_score >= self.metadata['ensemble_threshold']
        risk_level = self._get_risk_level(anomaly_score)
        risk_factors = self._calculate_risk_factors(input_data, derived, anomaly_score)
        
        return {
            "is_fraud": bool(is_fraud),
            "risk_level": risk_level,
            "anomaly_score": float(anomaly_score),
            "isolation_forest_score": float(iso_norm),
            "autoencoder_score": float(ae_norm),
            "risk_factors": risk_factors,
            "input_data": input_data,
            "derived_features": derived,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_explanation(self, result: Dict[str, Any]) -> str:
        """분석 결과에 대한 AI 설명을 생성한다."""
        prompt = f"""
FDS 분석 결과를 설명해달라.

거래 정보:
- 금액: ${result['input_data']['purchase_value']}
- 나이: {result['input_data']['age']}세
- 유입: {result['input_data']['source']}
- 시간차: {result['derived_features']['time_diff_hours']:.1f}시간

분석 결과:
- 판정: {'사기 의심' if result['is_fraud'] else '정상'}
- 위험도: {result['risk_level']}
- 점수: {result['anomaly_score']:.2%}
- 위험 요인: {', '.join(result['risk_factors']) if result['risk_factors'] else '없음'}

2-3문장으로 설명해달라. 문장은 ~다로 끝낸다.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content
    
    def generate_tool_definition(self) -> Dict[str, Any]:
        """OpenAI Function Calling을 위한 도구 정의를 생성한다."""
        return {
            "type": "function",
            "function": {
                "name": "analyze_transaction",
                "description": self.skill['description'],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "purchase_value": {"type": "number", "description": "구매 금액"},
                        "age": {"type": "integer", "description": "사용자 나이"},
                        "source": {"type": "string", "description": "유입 경로", "enum": ["SEO", "Ads", "Direct"]},
                        "browser": {"type": "string", "description": "브라우저", "enum": ["Chrome", "Safari", "FireFox", "IE", "Opera"]},
                        "sex": {"type": "string", "description": "성별", "enum": ["M", "F"]},
                        "signup_time": {"type": "string", "description": "가입 시간 (ISO format)"},
                        "purchase_time": {"type": "string", "description": "구매 시간 (ISO format)"}
                    },
                    "required": ["purchase_value", "age", "source", "browser", "sex", "signup_time", "purchase_time"]
                }
            }
        }


def create_skill_file(skill_path: str = 'skills/agent_skill.yaml'):
    """스킬 정의 파일을 생성한다."""
    skill_definition = {
        'agent_name': 'FraudDetectionAgent',
        'version': '1.0.0',
        'description': '거래 데이터를 분석하여 사기 가능성을 탐지하는 이상치 탐지 에이전트',
        
        'capabilities': {
            'primary': '거래의 사기 가능성을 실시간으로 탐지하고 위험도를 평가한다',
            'secondary': [
                '앙상블 이상치 점수 산출',
                '위험도 등급 분류 (LOW/MEDIUM/HIGH/CRITICAL)',
                '의심 요인 분석',
                'AI 기반 탐지 설명 생성'
            ]
        },
        
        'input_schema': {
            'type': 'object',
            'properties': {
                'purchase_value': {'type': 'number', 'minimum': 0},
                'age': {'type': 'integer', 'minimum': 0, 'maximum': 120},
                'source': {'type': 'string', 'enum': ['SEO', 'Ads', 'Direct']},
                'browser': {'type': 'string', 'enum': ['Chrome', 'Safari', 'FireFox', 'IE', 'Opera']},
                'sex': {'type': 'string', 'enum': ['M', 'F']},
                'signup_time': {'type': 'string'},
                'purchase_time': {'type': 'string'}
            },
            'required': ['purchase_value', 'age', 'source', 'browser', 'sex', 'signup_time', 'purchase_time']
        },
        
        'output_schema': {
            'type': 'object',
            'properties': {
                'is_fraud': {'type': 'boolean'},
                'risk_level': {'type': 'string'},
                'anomaly_score': {'type': 'number'},
                'risk_factors': {'type': 'array'}
            }
        },
        
        'model_info': {
            'isolation_forest_path': 'models/isolation_forest.pkl',
            'autoencoder_path': 'models/autoencoder.pth',
            'scaler_path': 'models/fds_scaler.pkl',
            'label_encoders_path': 'models/label_encoders.pkl',
            'metadata_path': 'models/fds_metadata.json'
        },
        
        'risk_thresholds': {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.85
        }
    }
    
    Path(skill_path).parent.mkdir(exist_ok=True)
    
    with open(skill_path, 'w', encoding='utf-8') as f:
        yaml.dump(skill_definition, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"스킬 파일 생성 완료: {skill_path}")
    return skill_path


if __name__ == '__main__':
    create_skill_file()
    agent = FDSEnableAgent('skills/agent_skill.yaml')
    print("\n" + agent.get_capability_description())
    
    # 테스트
    test_input = {
        'purchase_value': 45, 'age': 35, 'source': 'SEO', 'browser': 'Chrome', 'sex': 'M',
        'signup_time': '2024-01-15T10:30:00', 'purchase_time': '2024-02-20T14:25:00'
    }
    result = agent.analyze(test_input)
    print(f"\n예측 결과: {json.dumps(result, indent=2, ensure_ascii=False, default=str)}")
