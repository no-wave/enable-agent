#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Plastic Detection - 모델 학습 모듈

COCO 포맷 데이터를 YOLO 포맷으로 변환하고 YOLOv8 모델을 학습한다.

클래스:
- PET: 폴리에틸렌 테레프탈레이트 (#1)
- PS: 폴리스타이렌 (#6)
- PP: 폴리프로필렌 (#5)
- PE: 폴리에틸렌 (#2, #4)
"""

import json
import shutil
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import yaml
from ultralytics import YOLO


# 플라스틱 클래스 정의
PLASTIC_CLASSES = ['pet', 'ps', 'pp', 'pe']

# 클래스별 상세 정보
CLASS_INFO = {
    'pet': {
        'name': 'PET',
        'full_name': '폴리에틸렌 테레프탈레이트',
        'recycle_code': '#1',
        'examples': '음료수 병, 생수 병',
        'recyclable': True
    },
    'ps': {
        'name': 'PS',
        'full_name': '폴리스타이렌',
        'recycle_code': '#6',
        'examples': '요거트 컵, 스티로폼',
        'recyclable': False
    },
    'pp': {
        'name': 'PP',
        'full_name': '폴리프로필렌',
        'recycle_code': '#5',
        'examples': '식품 용기, 병뚜껑',
        'recyclable': True
    },
    'pe': {
        'name': 'PE',
        'full_name': '폴리에틸렌',
        'recycle_code': '#2, #4',
        'examples': '비닐봉지, 세제 용기',
        'recyclable': True
    }
}


def load_all_coco_annotations(annotations_dir: str) -> dict:
    """
    모든 COCO JSON 파일을 로드하고 통합한다.
    
    Args:
        annotations_dir: 어노테이션 디렉토리 경로
    
    Returns:
        dict: 통합된 COCO 데이터 (images, annotations, categories)
    """
    all_images = []
    all_annotations = []
    categories = None
    
    image_id_offset = 0
    annotation_id_offset = 0
    
    annotations_path = Path(annotations_dir)
    json_files = list(annotations_path.glob('*.json'))
    
    print(f"총 {len(json_files)}개의 JSON 파일 로드 중...")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 카테고리는 첫 번째 파일에서만 가져옴
        if categories is None:
            categories = coco_data.get('categories', [])
        
        # 이미지 ID 재매핑
        image_id_map = {}
        for img in coco_data.get('images', []):
            old_id = img['id']
            new_id = image_id_offset
            image_id_map[old_id] = new_id
            
            img['id'] = new_id
            all_images.append(img)
            image_id_offset += 1
        
        # 어노테이션 ID 및 image_id 재매핑
        for ann in coco_data.get('annotations', []):
            ann['id'] = annotation_id_offset
            ann['image_id'] = image_id_map.get(ann['image_id'], ann['image_id'])
            all_annotations.append(ann)
            annotation_id_offset += 1
    
    merged_coco = {
        'images': all_images,
        'annotations': all_annotations,
        'categories': categories or []
    }
    
    print(f"✓ 통합 완료:")
    print(f"  이미지: {len(all_images)}개")
    print(f"  어노테이션: {len(all_annotations)}개")
    print(f"  카테고리: {len(categories or [])}개")
    
    return merged_coco


def coco_to_yolo_bbox(bbox: list, img_width: int, img_height: int) -> list:
    """
    COCO bbox를 YOLO 포맷으로 변환한다.
    
    Args:
        bbox: [x, y, width, height] (COCO 포맷)
        img_width: 이미지 너비
        img_height: 이미지 높이
    
    Returns:
        list: [x_center, y_center, width, height] (YOLO 포맷, 정규화됨)
    """
    x, y, w, h = bbox
    
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center, y_center, w_norm, h_norm]


def create_yolo_dataset(coco_data: dict, train_ids: set, val_ids: set, 
                        output_base_dir: str, images_source_dir: str) -> tuple:
    """
    COCO 데이터를 YOLO 포맷으로 변환하고 train/val로 분할한다.
    
    Args:
        coco_data: COCO 어노테이션 데이터
        train_ids: 학습 이미지 ID 집합
        val_ids: 검증 이미지 ID 집합
        output_base_dir: 출력 베이스 디렉토리
        images_source_dir: 원본 이미지 디렉토리
    
    Returns:
        tuple: (train_count, val_count)
    """
    output_base_dir = Path(output_base_dir)
    images_source_dir = Path(images_source_dir)
    
    # 디렉토리 생성
    for split in ['train', 'val']:
        (output_base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_base_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 이미지 ID -> 이미지 정보 매핑
    images_info = {img['id']: img for img in coco_data['images']}
    
    # 이미지별 어노테이션 그룹화
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # 카테고리 ID -> 인덱스 매핑
    category_id_to_idx = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
    
    train_count = 0
    val_count = 0
    
    # 각 이미지 처리
    for img_id, img_info in images_info.items():
        # Train/Val 구분
        if img_id in train_ids:
            split = 'train'
            train_count += 1
        elif img_id in val_ids:
            split = 'val'
            val_count += 1
        else:
            continue
        
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 이미지 파일 복사
        src_img_path = images_source_dir / img_filename
        dst_img_path = output_base_dir / split / 'images' / img_filename
        
        if src_img_path.exists():
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"⚠️  이미지를 찾을 수 없음: {src_img_path}")
            continue
        
        # YOLO 라벨 생성
        label_filename = Path(img_filename).stem + '.txt'
        label_path = output_base_dir / split / 'labels' / label_filename
        
        yolo_annotations = []
        
        for ann in image_annotations[img_id]:
            bbox = ann['bbox']
            category_id = ann['category_id']
            
            class_idx = category_id_to_idx.get(category_id, 0)
            yolo_bbox = coco_to_yolo_bbox(bbox, img_width, img_height)
            
            yolo_line = f"{class_idx} {' '.join(map(str, yolo_bbox))}"
            yolo_annotations.append(yolo_line)
        
        # 라벨 파일 저장
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    print(f"\n=== 데이터셋 생성 완료 ===")
    print(f"Train: {train_count}개 (이미지 + 라벨)")
    print(f"Val: {val_count}개 (이미지 + 라벨)")
    
    return train_count, val_count


def prepare_yolo_dataset(raw_data_dir: str = 'data/raw_data', 
                         output_dir: str = 'datasets/plastic',
                         train_ratio: float = 0.8) -> dict:
    """
    원본 데이터에서 YOLO 데이터셋을 준비한다.
    
    Args:
        raw_data_dir: 원본 데이터 디렉토리
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율
    
    Returns:
        dict: 데이터셋 정보
    """
    print("=" * 60)
    print("YOLO 데이터셋 준비")
    print("=" * 60)
    
    raw_data_path = Path(raw_data_dir)
    annotations_dir = raw_data_path / 'annotations'
    images_dir = raw_data_path / 'images'
    
    # 데이터 존재 확인
    if not annotations_dir.exists() or not images_dir.exists():
        print(f"⚠️  데이터 디렉토리가 없다:")
        print(f"  annotations: {annotations_dir.exists()}")
        print(f"  images: {images_dir.exists()}")
        return None
    
    # COCO 어노테이션 로드
    coco_data = load_all_coco_annotations(str(annotations_dir))
    
    if not coco_data['images']:
        print("⚠️  이미지 데이터가 없다.")
        return None
    
    # 클래스 정보 추출
    classes = [cat['name'].lower() for cat in coco_data['categories']]
    print(f"\n클래스: {classes}")
    
    # Train/Val 분할
    all_image_ids = [img['id'] for img in coco_data['images']]
    random.seed(42)
    random.shuffle(all_image_ids)
    
    split_idx = int(len(all_image_ids) * train_ratio)
    train_image_ids = set(all_image_ids[:split_idx])
    val_image_ids = set(all_image_ids[split_idx:])
    
    print(f"\n=== Train/Val Split ===")
    print(f"Train: {len(train_image_ids)}개")
    print(f"Val: {len(val_image_ids)}개")
    
    # YOLO 데이터셋 생성
    train_cnt, val_cnt = create_yolo_dataset(
        coco_data,
        train_image_ids,
        val_image_ids,
        output_dir,
        str(images_dir)
    )
    
    # YAML 설정 파일 생성
    output_path = Path(output_dir)
    yaml_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = output_path.parent / 'plastic.yaml'
    yaml_path.parent.mkdir(exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"\n✓ YAML 설정 저장: {yaml_path}")
    
    return {
        'yaml_path': str(yaml_path),
        'classes': classes,
        'train_count': train_cnt,
        'val_count': val_cnt
    }


def train_model(data_yaml: str = 'datasets/plastic.yaml',
                model_name: str = 'yolov8n.pt',
                epochs: int = 50,
                imgsz: int = 640,
                batch: int = 16,
                patience: int = 10,
                model_dir: str = 'models') -> tuple:
    """
    YOLOv8 모델을 학습한다.
    
    Args:
        data_yaml: 데이터셋 YAML 경로
        model_name: 기본 모델 이름
        epochs: 학습 에폭 수
        imgsz: 입력 이미지 크기
        batch: 배치 크기
        patience: Early stopping patience
        model_dir: 모델 저장 디렉토리
    
    Returns:
        tuple: (model, results, metadata)
    """
    print("\n" + "=" * 60)
    print("YOLOv8 모델 학습")
    print("=" * 60)
    
    # 모델 로드
    model = YOLO(model_name)
    print(f"✓ 모델 로드: {model_name}")
    
    # 학습
    print(f"\n학습 시작 (epochs={epochs}, imgsz={imgsz}, batch={batch})")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        save=True,
        project='runs/detect',
        name='plastic_yolov8',
        exist_ok=True,
        seed=42
    )
    
    # 검증
    print("\n모델 검증 중...")
    val_results = model.val(data=data_yaml, verbose=False)
    
    # 메트릭 출력
    print("\n=== 검증 결과 ===")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")
    
    # 모델 저장
    Path(model_dir).mkdir(exist_ok=True)
    
    # Best 모델 복사
    src_model = Path('runs/detect/plastic_yolov8/weights/best.pt')
    if src_model.exists():
        dst_model = Path(model_dir) / 'plastic_yolov8_best.pt'
        shutil.copy(src_model, dst_model)
        print(f"\n✓ 모델 저장: {dst_model}")
    
    # 메타데이터 저장
    metadata = {
        'model_name': 'plastic_yolov8',
        'base_model': model_name,
        'classes': PLASTIC_CLASSES,
        'class_info': CLASS_INFO,
        'num_classes': len(PLASTIC_CLASSES),
        'input_size': imgsz,
        'trained_at': datetime.now().isoformat(),
        'metrics': {
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr)
        }
    }
    
    metadata_path = Path(model_dir) / 'yolo_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 메타데이터 저장: {metadata_path}")
    
    return model, val_results, metadata


if __name__ == '__main__':
    # 데이터셋 준비
    dataset_info = prepare_yolo_dataset()
    
    if dataset_info:
        # 모델 학습
        model, results, metadata = train_model(
            data_yaml=dataset_info['yaml_path'],
            epochs=50
        )
        
        print("\n" + "=" * 60)
        print("학습 완료!")
        print("=" * 60)
