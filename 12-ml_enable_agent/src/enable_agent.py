#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iris Classification Enable Agent 구현 모듈

Random Forest 모델을 Enable Agent로 래핑하여 LLM 통합을 지원한다.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

import numpy as np
import joblib
from openai import OpenAI


# Iris 클래스 정보
CLASS_INFO = {
    'setosa': {
        'korean_name': '부채붓꽃',
        'description': '꽃잎이 작고 꽃받침이 넓은 품종이다.',
        'characteristics': '꽃잎 길이 < 2cm, 꽃잎 너비 < 0.5cm',
        'color': '보라색, 흰색'
    },
    'versicolor': {
        'korean_name': '버시컬러 붓꽃',
        'description': '중간 크기의 꽃잎을 가진 품종이다.',
        'characteristics': '꽃잎 길이 3-5cm, 꽃잎 너비 1-1.5cm',
        'color': '파란색, 보라색'
    },
    'virginica': {
        'korean_name': '버지니카 붓꽃',
        'description': '꽃잎이 크고 길쭉한 품종이다.',
        'characteristics': '꽃잎 길이 > 5cm, 꽃잎 너비 > 1.5cm',
        'color': '파란색, 보라색'
    }
}


class IrisClassificationAgent:
    """Iris 분류 Enable Agent"""
    
    def __init__(self, skill_path: str):
        """
        Agent를 초기화한다.
        
        Args:
            skill_path: 스킬 정의 YAML 파일 경로
        """
        # 스킬 정의 로드
        with open(skill_path, 'r', encoding='utf-8') as f:
            self.skill = yaml.safe_load(f)
        
        # 모델 로드
        model_path = self.skill['model_info']['model_path']
        self.model = joblib.load(model_path)
        
        # 메타데이터 로드
        metadata_path = self.skill['model_info'].get('metadata_path')
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # 특성 및 클래스 정보
        self.feature_names = self.metadata.get('feature_names', [
            'sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)'
        ])
        self.target_names = self.metadata.get('target_names', ['setosa', 'versicolor', 'virginica'])
        self.class_info = CLASS_INFO
        
        # OpenAI 클라이언트
        self.client = OpenAI()
        
        print(f"✓ {self.skill['agent_name']} 초기화 완료")
        print(f"  모델: {model_path}")
        print(f"  정확도: {self.metadata.get('accuracy', 'N/A'):.2%}" if isinstance(self.metadata.get('accuracy'), float) else f"  정확도: N/A")
    
    def predict(self, features: Union[Dict, List, np.ndarray]) -> Dict[str, Any]:
        """
        붓꽃 품종을 예측한다.
        
        Args:
            features: 입력 특성 (dict, list, 또는 numpy array)
                - dict: {'sepal_length': 5.1, 'sepal_width': 3.5, ...}
                - list: [5.1, 3.5, 1.4, 0.2]
                - ndarray: np.array([5.1, 3.5, 1.4, 0.2])
        
        Returns:
            dict: 예측 결과
        """
        # 입력 정규화
        if isinstance(features, dict):
            # dict -> list 변환
            feature_values = []
            for name in self.feature_names:
                # 다양한 키 형식 지원
                key_variants = [
                    name,
                    name.replace(' ', '_').replace('(', '').replace(')', ''),
                    name.split(' ')[0] + '_' + name.split(' ')[1] if len(name.split(' ')) > 1 else name
                ]
                
                value = None
                for key in key_variants:
                    if key in features:
                        value = features[key]
                        break
                    # 대소문자 무시
                    for k, v in features.items():
                        if k.lower().replace(' ', '_') == key.lower().replace(' ', '_'):
                            value = v
                            break
                
                if value is None:
                    raise ValueError(f"특성 '{name}'을 찾을 수 없다.")
                feature_values.append(float(value))
            
            X = np.array([feature_values])
        elif isinstance(features, list):
            X = np.array([features])
        else:
            X = np.array(features).reshape(1, -1)
        
        # 예측
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # 예측 결과
        predicted_class = self.target_names[prediction]
        confidence = float(probabilities[prediction])
        
        # 클래스별 확률
        class_probabilities = {
            name: float(prob) for name, prob in zip(self.target_names, probabilities)
        }
        
        # 클래스 정보
        class_detail = self.class_info.get(predicted_class, {})
        
        result = {
            'predicted_class': predicted_class,
            'korean_name': class_detail.get('korean_name', predicted_class),
            'confidence': confidence,
            'probabilities': class_probabilities,
            'description': class_detail.get('description', ''),
            'characteristics': class_detail.get('characteristics', ''),
            'input_features': {
                name: float(X[0][i]) for i, name in enumerate(self.feature_names)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, samples: List[Union[Dict, List]]) -> List[Dict]:
        """
        여러 샘플을 한번에 예측한다.
        
        Args:
            samples: 샘플 리스트
        
        Returns:
            list: 예측 결과 리스트
        """
        results = []
        for sample in samples:
            result = self.predict(sample)
            results.append(result)
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """특성 중요도를 반환한다."""
        return self.metadata.get('feature_importance', {})
    
    def generate_explanation(self, prediction_result: Dict) -> str:
        """
        AI를 사용하여 예측 결과에 대한 설명을 생성한다.
        
        Args:
            prediction_result: predict() 메서드의 결과
        
        Returns:
            str: AI 생성 설명
        """
        features = prediction_result['input_features']
        predicted = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        probs = prediction_result['probabilities']
        
        prompt = f"""붓꽃 분류 결과를 설명해달라.

입력 특성:
- 꽃받침 길이: {features.get('sepal length (cm)', 0):.1f}cm
- 꽃받침 너비: {features.get('sepal width (cm)', 0):.1f}cm
- 꽃잎 길이: {features.get('petal length (cm)', 0):.1f}cm
- 꽃잎 너비: {features.get('petal width (cm)', 0):.1f}cm

예측 결과: {predicted} ({prediction_result['korean_name']})
신뢰도: {confidence:.1%}

클래스별 확률:
- Setosa: {probs['setosa']:.1%}
- Versicolor: {probs['versicolor']:.1%}
- Virginica: {probs['virginica']:.1%}

특성 중요도: {self.get_feature_importance()}

왜 이 품종으로 예측되었는지 2-3문장으로 설명하라. 문장은 ~다로 끝낸다."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def generate_tool_definition(self) -> Dict:
        """
        OpenAI Function Calling을 위한 도구 정의를 생성한다.
        
        Returns:
            dict: 도구 정의
        """
        return {
            "type": "function",
            "function": {
                "name": "predict",
                "description": self.skill['description'],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sepal_length": {
                            "type": "number",
                            "description": "꽃받침 길이 (cm)"
                        },
                        "sepal_width": {
                            "type": "number",
                            "description": "꽃받침 너비 (cm)"
                        },
                        "petal_length": {
                            "type": "number",
                            "description": "꽃잎 길이 (cm)"
                        },
                        "petal_width": {
                            "type": "number",
                            "description": "꽃잎 너비 (cm)"
                        }
                    },
                    "required": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
                }
            }
        }


def create_skill_file(skill_path: str = 'skills/agent_skill.yaml') -> str:
    """
    Enable Agent 스킬 정의 파일을 생성한다.
    
    Args:
        skill_path: 스킬 파일 경로
    
    Returns:
        str: 생성된 스킬 파일 경로
    """
    skill = {
        'agent_name': 'IrisClassificationAgent',
        'version': '1.0.0',
        'description': '붓꽃(Iris) 품종을 분류하는 에이전트. 꽃받침과 꽃잎의 길이/너비를 입력받아 Setosa, Versicolor, Virginica 중 하나로 분류한다.',
        
        'capabilities': {
            'primary': '붓꽃 품종 분류',
            'secondary': [
                '분류 결과 설명',
                '특성 중요도 분석',
                '신뢰도 기반 예측'
            ]
        },
        
        'classes': {
            'setosa': CLASS_INFO['setosa'],
            'versicolor': CLASS_INFO['versicolor'],
            'virginica': CLASS_INFO['virginica']
        },
        
        'features': {
            'sepal_length': {
                'name': 'sepal length (cm)',
                'korean_name': '꽃받침 길이',
                'unit': 'cm',
                'typical_range': [4.3, 7.9]
            },
            'sepal_width': {
                'name': 'sepal width (cm)',
                'korean_name': '꽃받침 너비',
                'unit': 'cm',
                'typical_range': [2.0, 4.4]
            },
            'petal_length': {
                'name': 'petal length (cm)',
                'korean_name': '꽃잎 길이',
                'unit': 'cm',
                'typical_range': [1.0, 6.9]
            },
            'petal_width': {
                'name': 'petal width (cm)',
                'korean_name': '꽃잎 너비',
                'unit': 'cm',
                'typical_range': [0.1, 2.5]
            }
        },
        
        'input_schema': {
            'type': 'object',
            'properties': {
                'sepal_length': {'type': 'number', 'description': '꽃받침 길이 (cm)'},
                'sepal_width': {'type': 'number', 'description': '꽃받침 너비 (cm)'},
                'petal_length': {'type': 'number', 'description': '꽃잎 길이 (cm)'},
                'petal_width': {'type': 'number', 'description': '꽃잎 너비 (cm)'}
            },
            'required': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        },
        
        'output_schema': {
            'type': 'object',
            'properties': {
                'predicted_class': {'type': 'string', 'description': '예측된 품종'},
                'korean_name': {'type': 'string', 'description': '품종 한글명'},
                'confidence': {'type': 'number', 'description': '예측 신뢰도'},
                'probabilities': {'type': 'object', 'description': '클래스별 확률'}
            }
        },
        
        'model_info': {
            'model_path': 'models/iris_classifier.pkl',
            'metadata_path': 'models/iris_classifier_metadata.json',
            'model_type': 'RandomForestClassifier'
        }
    }
    
    # 디렉토리 생성
    Path(skill_path).parent.mkdir(parents=True, exist_ok=True)
    
    # YAML 저장
    with open(skill_path, 'w', encoding='utf-8') as f:
        yaml.dump(skill, allow_unicode=True, stream=f, sort_keys=False, default_flow_style=False)
    
    print(f"✓ 스킬 파일 생성: {skill_path}")
    
    return skill_path


if __name__ == '__main__':
    # 스킬 파일 생성
    skill_path = create_skill_file()
    
    # Agent 테스트
    agent = IrisClassificationAgent(skill_path)
    
    # 테스트 예측
    test_sample = {
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
    
    result = agent.predict(test_sample)
    print("\n=== 예측 결과 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
