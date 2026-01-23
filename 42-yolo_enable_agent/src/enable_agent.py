#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Plastic Detection Enable Agent 구현 모듈

YOLOv8 모델을 Enable Agent로 래핑하여 LLM 통합을 지원한다.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from ultralytics import YOLO
from openai import OpenAI


# 플라스틱 클래스 정보
CLASS_INFO = {
    0: {
        'name': 'PET',
        'full_name': '폴리에틸렌 테레프탈레이트',
        'recycle_code': '#1',
        'examples': '음료수 병, 생수 병',
        'recyclable': True,
        'disposal_guide': '라벨을 제거하고 내용물을 비운 후 압축하여 배출한다.'
    },
    1: {
        'name': 'PS',
        'full_name': '폴리스타이렌',
        'recycle_code': '#6',
        'examples': '요거트 컵, 스티로폼',
        'recyclable': False,
        'disposal_guide': '일반 쓰레기로 배출한다. 스티로폼은 별도 배출한다.'
    },
    2: {
        'name': 'PP',
        'full_name': '폴리프로필렌',
        'recycle_code': '#5',
        'examples': '식품 용기, 병뚜껑',
        'recyclable': True,
        'disposal_guide': '깨끗이 세척 후 플라스틱류로 배출한다.'
    },
    3: {
        'name': 'PE',
        'full_name': '폴리에틸렌',
        'recycle_code': '#2, #4',
        'examples': '비닐봉지, 세제 용기',
        'recyclable': True,
        'disposal_guide': 'HDPE는 플라스틱류, LDPE(비닐)는 비닐류로 배출한다.'
    }
}


class PlasticDetectionAgent:
    """플라스틱 객체 탐지 Enable Agent"""
    
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
        if Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # 기본 모델 사용
            print(f"⚠️  모델 파일 없음: {model_path}, 기본 모델 사용")
            self.model = YOLO('yolov8n.pt')
        
        # 메타데이터 로드
        metadata_path = self.skill['model_info'].get('metadata_path')
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # 클래스 정보 설정
        self.classes = self.skill.get('classes', CLASS_INFO)
        if isinstance(self.classes, list):
            # 리스트를 딕셔너리로 변환
            self.classes = {i: CLASS_INFO.get(i, {'name': c}) for i, c in enumerate(self.classes)}
        
        # OpenAI 클라이언트
        self.client = OpenAI()
        
        print(f"✓ {self.skill['agent_name']} 초기화 완료")
        print(f"  모델: {model_path}")
        print(f"  클래스: {[self.classes[i]['name'] for i in sorted(self.classes.keys())]}")
    
    def detect(self, image_path: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        이미지에서 플라스틱 객체를 탐지한다.
        
        Args:
            image_path: 이미지 파일 경로
            confidence_threshold: 신뢰도 임계값
        
        Returns:
            dict: 탐지 결과
        """
        # 이미지 경로 확인
        if not Path(image_path).exists():
            return {
                'error': f'이미지를 찾을 수 없다: {image_path}',
                'detections': [],
                'num_detections': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        # YOLO 추론
        results = self.model.predict(
            source=image_path,
            conf=confidence_threshold,
            verbose=False
        )
        
        detections = []
        class_counts = {}
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # 클래스 정보
                class_info = self.classes.get(cls_id, {'name': f'class_{cls_id}'})
                
                detection = {
                    'class_id': cls_id,
                    'class_name': class_info.get('name', f'class_{cls_id}'),
                    'full_name': class_info.get('full_name', ''),
                    'recycle_code': class_info.get('recycle_code', ''),
                    'recyclable': class_info.get('recyclable', False),
                    'confidence': confidence,
                    'bbox': bbox,
                    'disposal_guide': class_info.get('disposal_guide', '')
                }
                detections.append(detection)
                
                # 클래스별 카운트
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'image_path': str(image_path),
            'detections': detections,
            'num_detections': len(detections),
            'class_counts': class_counts,
            'recyclable_count': sum(1 for d in detections if d['recyclable']),
            'non_recyclable_count': sum(1 for d in detections if not d['recyclable']),
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_batch(self, image_paths: List[str], confidence_threshold: float = 0.25) -> List[Dict]:
        """
        여러 이미지에서 플라스틱 객체를 탐지한다.
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            confidence_threshold: 신뢰도 임계값
        
        Returns:
            list: 탐지 결과 리스트
        """
        results = []
        for path in image_paths:
            result = self.detect(path, confidence_threshold)
            results.append(result)
        return results
    
    def get_recycling_guide(self, detection_result: Dict) -> str:
        """
        탐지 결과에 대한 재활용 가이드를 생성한다.
        
        Args:
            detection_result: detect() 메서드의 결과
        
        Returns:
            str: 재활용 가이드 텍스트
        """
        if not detection_result.get('detections'):
            return "탐지된 플라스틱이 없다."
        
        guide_lines = ["=== 재활용 가이드 ===\n"]
        
        for det in detection_result['detections']:
            recyclable_status = "♻️ 재활용 가능" if det['recyclable'] else "❌ 재활용 불가"
            guide_lines.append(f"[{det['class_name']}] {recyclable_status}")
            guide_lines.append(f"  - {det['full_name']} ({det['recycle_code']})")
            guide_lines.append(f"  - 배출 방법: {det['disposal_guide']}")
            guide_lines.append("")
        
        # 요약
        guide_lines.append(f"총 {detection_result['num_detections']}개 탐지")
        guide_lines.append(f"재활용 가능: {detection_result['recyclable_count']}개")
        guide_lines.append(f"재활용 불가: {detection_result['non_recyclable_count']}개")
        
        return "\n".join(guide_lines)
    
    def generate_explanation(self, detection_result: Dict) -> str:
        """
        AI를 사용하여 탐지 결과에 대한 설명을 생성한다.
        
        Args:
            detection_result: detect() 메서드의 결과
        
        Returns:
            str: AI 생성 설명
        """
        if not detection_result.get('detections'):
            return "이미지에서 플라스틱 객체가 탐지되지 않았다."
        
        # 탐지 요약
        summary = []
        for det in detection_result['detections']:
            summary.append(f"{det['class_name']}({det['confidence']:.0%})")
        
        prompt = f"""플라스틱 객체 탐지 결과를 설명해달라.

탐지된 객체: {', '.join(summary)}
총 개수: {detection_result['num_detections']}개
재활용 가능: {detection_result['recyclable_count']}개
재활용 불가: {detection_result['non_recyclable_count']}개

각 플라스틱 종류의 특성과 올바른 분리수거 방법을 2-3문장으로 설명하라. 문장은 ~다로 끝낸다."""

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
                "name": "detect",
                "description": self.skill['description'],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "탐지할 이미지 파일 경로"
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "description": "신뢰도 임계값 (0-1)",
                            "default": 0.25
                        }
                    },
                    "required": ["image_path"]
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
        'agent_name': 'PlasticDetectionAgent',
        'version': '1.0.0',
        'description': '플라스틱 재활용 분류를 위한 객체 탐지 에이전트. 이미지에서 PET, PS, PP, PE 플라스틱을 탐지하고 재활용 가이드를 제공한다.',
        
        'capabilities': {
            'primary': '플라스틱 객체 탐지',
            'secondary': [
                '재활용 분류 가이드',
                '배출 방법 안내',
                '탐지 결과 시각화'
            ]
        },
        
        'classes': {
            0: CLASS_INFO[0],
            1: CLASS_INFO[1],
            2: CLASS_INFO[2],
            3: CLASS_INFO[3]
        },
        
        'input_schema': {
            'type': 'object',
            'properties': {
                'image_path': {
                    'type': 'string',
                    'description': '탐지할 이미지 경로'
                },
                'confidence_threshold': {
                    'type': 'number',
                    'description': '신뢰도 임계값',
                    'default': 0.25
                }
            },
            'required': ['image_path']
        },
        
        'output_schema': {
            'type': 'object',
            'properties': {
                'detections': {
                    'type': 'array',
                    'description': '탐지된 객체 리스트'
                },
                'num_detections': {
                    'type': 'integer',
                    'description': '탐지된 객체 수'
                },
                'class_counts': {
                    'type': 'object',
                    'description': '클래스별 탐지 수'
                },
                'recyclable_count': {
                    'type': 'integer',
                    'description': '재활용 가능 객체 수'
                }
            }
        },
        
        'model_info': {
            'model_path': 'models/plastic_yolov8_best.pt',
            'metadata_path': 'models/yolo_metadata.json',
            'model_type': 'YOLOv8'
        }
    }
    
    # 디렉토리 생성
    Path(skill_path).parent.mkdir(parents=True, exist_ok=True)
    
    # YAML 저장
    with open(skill_path, 'w', encoding='utf-8') as f:
        yaml.dump(skill, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    print(f"✓ 스킬 파일 생성: {skill_path}")
    
    return skill_path


if __name__ == '__main__':
    # 스킬 파일 생성
    skill_path = create_skill_file()
    
    # Agent 테스트
    agent = PlasticDetectionAgent(skill_path)
    
    # 테스트 이미지가 있으면 탐지 실행
    test_images = list(Path('datasets/plastic/val/images').glob('*.jpg'))
    if test_images:
        result = agent.detect(str(test_images[0]))
        print("\n=== 탐지 결과 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        
        print("\n" + agent.get_recycling_guide(result))
