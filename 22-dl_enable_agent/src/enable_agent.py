"""
Enable Agent 구현 모듈

PyTorch 딥러닝 모델을 Enable Agent로 래핑하여 LLM 기반 시스템에 통합한다.
"""

import json
import yaml
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from openai import OpenAI


class SpaceshipClassifier(nn.Module):
    """Spaceship Titanic 이진 분류 신경망"""
    
    def __init__(self, input_dim: int = 10, hidden_dims: list = [64, 32, 16], dropout_rate: float = 0.3):
        super(SpaceshipClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class SpaceshipEnableAgent:
    """Spaceship Titanic 예측 모델을 Enable Agent로 래핑한 클래스"""
    
    def __init__(self, skill_path: str):
        """Enable Agent를 초기화한다."""
        with open(skill_path, 'r', encoding='utf-8') as f:
            self.skill = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        model_path = self.skill['model_info']['model_path']
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        self.model = SpaceshipClassifier(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            dropout_rate=checkpoint['dropout_rate']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 스케일러 로드
        scaler_path = self.skill['model_info']['scaler_path']
        self.scaler = joblib.load(scaler_path)
        
        # 메타데이터 로드
        metadata_path = self.skill['model_info']['metadata_path']
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 인코딩 맵
        self.encoding_maps = self.skill['encoding_maps']
        
        # OpenAI 클라이언트
        self.client = OpenAI()
        
        print(f"Enable Agent 초기화 완료: {self.skill['agent_name']}")
        print(f"모델 정확도: {self.metadata['test_accuracy']:.4f}")
        print(f"디바이스: {self.device}")
    
    def get_capability_description(self) -> str:
        """에이전트의 기능을 자연어로 반환한다."""
        return f"""
Agent 이름: {self.skill['agent_name']}
버전: {self.skill['version']}
설명: {self.skill['description']}

주요 기능: {self.skill['capabilities']['primary']}
부가 기능: {', '.join(self.skill['capabilities']['secondary'])}

입력 파라미터:
- HomePlanet: 출발 행성 (0: Earth, 1: Europa, 2: Mars, 3: Unknown)
- CryoSleep: 냉동 수면 여부 (0: No, 1: Yes)
- Destination: 목적지 (0: TRAPPIST-1e, 1: PSO J318.5-22, 2: 55 Cancri e)
- Age: 나이
- VIP: VIP 서비스 여부 (0: No, 1: Yes)
- RoomService, FoodCourt, ShoppingMall, Spa, VRDeck: 각 서비스 지출액

출력:
- predicted_class: 예측된 이동 여부 (Transported / Not Transported)
- probability: 이동될 확률 (0-1)
- confidence: 예측 신뢰도 (0-1)

모델 정보:
- 타입: {self.metadata['model_type']}
- 아키텍처: {self.metadata['architecture']}
- 테스트 정확도: {self.metadata['test_accuracy']:.4f}
- AUC-ROC: {self.metadata['test_auc_roc']:.4f}
""".strip()
    
    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """입력 데이터의 유효성을 검증한다."""
        required_fields = self.skill['input_schema']['required']
        properties = self.skill['input_schema']['properties']
        
        for field in required_fields:
            if field not in input_data:
                return False, f"필수 필드 누락: {field}"
        
        for field, value in input_data.items():
            if field in properties:
                prop = properties[field]
                if 'minimum' in prop and value < prop['minimum']:
                    return False, f"{field}가 최소값({prop['minimum']})보다 작다: {value}"
                if 'maximum' in prop and value > prop['maximum']:
                    return False, f"{field}가 최대값({prop['maximum']})보다 크다: {value}"
        
        return True, "유효한 입력"
    
    def _decode_categorical(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """범주형 변수를 사람이 읽을 수 있는 형태로 변환한다."""
        decoded = {}
        for field, value in input_data.items():
            if field in self.encoding_maps:
                decoded[field] = self.encoding_maps[field].get(value, str(value))
            else:
                decoded[field] = value
        return decoded
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """예측을 수행한다."""
        is_valid, message = self.validate_input(input_data)
        if not is_valid:
            raise ValueError(message)
        
        feature_names = self.metadata['feature_names']
        input_array = np.array([[input_data[f] for f in feature_names]])
        
        input_scaled = self.scaler.transform(input_array)
        input_tensor = torch.FloatTensor(input_scaled).to(self.device)
        
        with torch.no_grad():
            probability = self.model(input_tensor).item()
        
        predicted_class = 'Transported' if probability > 0.5 else 'Not Transported'
        confidence = probability if probability > 0.5 else (1 - probability)
        
        decoded_features = self._decode_categorical(input_data)
        
        return {
            "predicted_class": predicted_class,
            "probability": float(probability),
            "confidence": float(confidence),
            "input_features": input_data,
            "decoded_features": decoded_features,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_explanation(self, prediction_result: Dict[str, Any]) -> str:
        """예측 결과에 대한 AI 설명을 생성한다."""
        decoded = prediction_result['decoded_features']
        
        prompt = f"""
Spaceship Titanic 승객의 다른 차원 이동 예측 결과를 분석하고 설명해달라.

## 승객 정보
- 출발 행성: {decoded['HomePlanet']}
- 냉동 수면: {decoded['CryoSleep']}
- 목적지: {decoded['Destination']}
- 나이: {decoded['Age']}세
- VIP: {decoded['VIP']}
- 룸서비스 지출: {decoded['RoomService']}
- 푸드코트 지출: {decoded['FoodCourt']}
- 쇼핑몰 지출: {decoded['ShoppingMall']}
- 스파 지출: {decoded['Spa']}
- VR 데크 지출: {decoded['VRDeck']}

## 예측 결과
- 예측: {prediction_result['predicted_class']}
- 이동 확률: {prediction_result['probability']:.2%}
- 신뢰도: {prediction_result['confidence']:.2%}

이 예측 결과가 왜 나왔는지 승객의 특성을 바탕으로 2-3문장으로 간략히 설명해달라.
특히 냉동 수면 여부, 지출 패턴, 출발 행성이 예측에 어떤 영향을 미쳤는지 분석해달라.
문장의 끝은 ~다로 끝낸다.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def generate_tool_definition(self) -> Dict[str, Any]:
        """OpenAI Function Calling을 위한 도구 정의를 생성한다."""
        return {
            "type": "function",
            "function": {
                "name": "predict_spaceship_transport",
                "description": self.skill['description'],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "HomePlanet": {
                            "type": "integer",
                            "description": "출발 행성 (0: Earth, 1: Europa, 2: Mars, 3: Unknown)"
                        },
                        "CryoSleep": {
                            "type": "integer",
                            "description": "냉동 수면 여부 (0: No, 1: Yes)"
                        },
                        "Destination": {
                            "type": "integer",
                            "description": "목적지 (0: TRAPPIST-1e, 1: PSO J318.5-22, 2: 55 Cancri e)"
                        },
                        "Age": {
                            "type": "number",
                            "description": "나이"
                        },
                        "VIP": {
                            "type": "integer",
                            "description": "VIP 서비스 여부 (0: No, 1: Yes)"
                        },
                        "RoomService": {
                            "type": "number",
                            "description": "룸서비스 지출액"
                        },
                        "FoodCourt": {
                            "type": "number",
                            "description": "푸드코트 지출액"
                        },
                        "ShoppingMall": {
                            "type": "number",
                            "description": "쇼핑몰 지출액"
                        },
                        "Spa": {
                            "type": "number",
                            "description": "스파 지출액"
                        },
                        "VRDeck": {
                            "type": "number",
                            "description": "VR 데크 지출액"
                        }
                    },
                    "required": ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
                                 "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
                }
            }
        }


def create_skill_file(skill_path: str = 'skills/agent_skill.yaml'):
    """스킬 정의 파일을 생성한다."""
    skill_definition = {
        'agent_name': 'SpaceshipTransportPredictor',
        'version': '1.0.0',
        'description': 'Spaceship Titanic 승객의 다른 차원 이동 여부를 예측하는 딥러닝 에이전트',
        
        'capabilities': {
            'primary': '승객 정보를 기반으로 다른 차원 이동(Transported) 여부를 예측한다',
            'secondary': [
                '예측 확률 및 신뢰도 제공',
                '입력 데이터 유효성 검증',
                'AI 기반 예측 설명 생성'
            ]
        },
        
        'input_schema': {
            'type': 'object',
            'properties': {
                'HomePlanet': {'type': 'integer', 'description': '출발 행성', 'minimum': 0, 'maximum': 3},
                'CryoSleep': {'type': 'integer', 'description': '냉동 수면 여부', 'minimum': 0, 'maximum': 1},
                'Destination': {'type': 'integer', 'description': '목적지', 'minimum': 0, 'maximum': 2},
                'Age': {'type': 'number', 'description': '나이', 'minimum': 0, 'maximum': 100},
                'VIP': {'type': 'integer', 'description': 'VIP 서비스 여부', 'minimum': 0, 'maximum': 1},
                'RoomService': {'type': 'number', 'description': '룸서비스 지출액', 'minimum': 0},
                'FoodCourt': {'type': 'number', 'description': '푸드코트 지출액', 'minimum': 0},
                'ShoppingMall': {'type': 'number', 'description': '쇼핑몰 지출액', 'minimum': 0},
                'Spa': {'type': 'number', 'description': '스파 지출액', 'minimum': 0},
                'VRDeck': {'type': 'number', 'description': 'VR 데크 지출액', 'minimum': 0}
            },
            'required': ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
                         'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        },
        
        'output_schema': {
            'type': 'object',
            'properties': {
                'predicted_class': {'type': 'string', 'description': '예측된 이동 여부'},
                'probability': {'type': 'number', 'description': '이동될 확률'},
                'confidence': {'type': 'number', 'description': '예측 신뢰도'},
                'input_features': {'type': 'object', 'description': '입력된 특성 값'},
                'timestamp': {'type': 'string', 'description': '예측 수행 시각'}
            }
        },
        
        'model_info': {
            'model_path': 'models/spaceship_classifier.pth',
            'scaler_path': 'models/spaceship_scaler.pkl',
            'metadata_path': 'models/spaceship_classifier_metadata.json',
            'framework': 'PyTorch'
        },
        
        'encoding_maps': {
            'HomePlanet': {0: 'Earth', 1: 'Europa', 2: 'Mars', 3: 'Unknown'},
            'Destination': {0: 'TRAPPIST-1e', 1: 'PSO J318.5-22', 2: '55 Cancri e'},
            'CryoSleep': {0: 'No', 1: 'Yes'},
            'VIP': {0: 'No', 1: 'Yes'}
        }
    }
    
    Path(skill_path).parent.mkdir(exist_ok=True)
    
    with open(skill_path, 'w', encoding='utf-8') as f:
        yaml.dump(skill_definition, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"스킬 파일 생성 완료: {skill_path}")
    return skill_path


if __name__ == '__main__':
    create_skill_file()
    agent = SpaceshipEnableAgent('skills/agent_skill.yaml')
    print("\n" + agent.get_capability_description())
    
    # 테스트 예측
    test_input = {
        'HomePlanet': 0, 'CryoSleep': 1, 'Destination': 2, 'Age': 28.0, 'VIP': 0,
        'RoomService': 0.0, 'FoodCourt': 0.0, 'ShoppingMall': 0.0, 'Spa': 0.0, 'VRDeck': 0.0
    }
    result = agent.predict(test_input)
    print(f"\n예측 결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
