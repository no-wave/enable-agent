#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iris Classification - 모델 학습 모듈

Iris 데이터셋으로 Random Forest 분류 모델을 학습한다.

클래스:
- Setosa: 부채붓꽃
- Versicolor: 버시컬러 붓꽃
- Virginica: 버지니카 붓꽃

특성:
- sepal length: 꽃받침 길이 (cm)
- sepal width: 꽃받침 너비 (cm)
- petal length: 꽃잎 길이 (cm)
- petal width: 꽃잎 너비 (cm)
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


# Iris 클래스 정보
CLASS_INFO = {
    'setosa': {
        'korean_name': '부채붓꽃',
        'description': '꽃잎이 작고 꽃받침이 넓은 품종이다.',
        'characteristics': '꽃잎 길이 < 2cm, 꽃잎 너비 < 0.5cm'
    },
    'versicolor': {
        'korean_name': '버시컬러 붓꽃',
        'description': '중간 크기의 꽃잎을 가진 품종이다.',
        'characteristics': '꽃잎 길이 3-5cm, 꽃잎 너비 1-1.5cm'
    },
    'virginica': {
        'korean_name': '버지니카 붓꽃',
        'description': '꽃잎이 크고 길쭉한 품종이다.',
        'characteristics': '꽃잎 길이 > 5cm, 꽃잎 너비 > 1.5cm'
    }
}

# 특성 정보
FEATURE_INFO = {
    'sepal length (cm)': {
        'korean_name': '꽃받침 길이',
        'unit': 'cm',
        'typical_range': '4.3 ~ 7.9'
    },
    'sepal width (cm)': {
        'korean_name': '꽃받침 너비',
        'unit': 'cm',
        'typical_range': '2.0 ~ 4.4'
    },
    'petal length (cm)': {
        'korean_name': '꽃잎 길이',
        'unit': 'cm',
        'typical_range': '1.0 ~ 6.9'
    },
    'petal width (cm)': {
        'korean_name': '꽃잎 너비',
        'unit': 'cm',
        'typical_range': '0.1 ~ 2.5'
    }
}


def load_iris_data():
    """
    Iris 데이터셋을 로드한다.
    
    Returns:
        tuple: (X, y, feature_names, target_names, df)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)
    
    # DataFrame 생성
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    
    print("=== Iris 데이터셋 로드 완료 ===")
    print(f"샘플 수: {len(X)}")
    print(f"특성 수: {len(feature_names)}")
    print(f"클래스: {target_names}")
    
    return X, y, feature_names, target_names, df


def train_model(model_dir: str = 'models',
                n_estimators: int = 100,
                max_depth: int = 5,
                random_state: int = 42,
                test_size: float = 0.2):
    """
    Random Forest 분류 모델을 학습한다.
    
    Args:
        model_dir: 모델 저장 디렉토리
        n_estimators: 트리 개수
        max_depth: 최대 깊이
        random_state: 랜덤 시드
        test_size: 테스트 세트 비율
    
    Returns:
        tuple: (model, accuracy, metadata)
    """
    print("\n" + "=" * 60)
    print("Iris Random Forest 모델 학습")
    print("=" * 60)
    
    # 데이터 로드
    X, y, feature_names, target_names, df = load_iris_data()
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n학습 세트: {len(X_train)}개")
    print(f"테스트 세트: {len(X_test)}개")
    
    # 모델 학습
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    print(f"\n✓ 모델 학습 완료")
    
    # 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== 모델 성능 ===")
    print(f"정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n분류 보고서:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 교차 검증
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\n5-Fold 교차 검증:")
    print(f"  평균 정확도: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    
    # 특성 중요도
    feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    print(f"\n특성 중요도:")
    for name, importance in sorted(feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {name}: {importance:.4f}")
    
    # 모델 저장
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    model_path = Path(model_dir) / 'iris_classifier.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ 모델 저장: {model_path}")
    
    # 메타데이터 저장
    metadata = {
        'model_name': 'Iris Random Forest Classifier',
        'model_type': 'RandomForestClassifier',
        'accuracy': float(accuracy),
        'cv_mean_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'feature_names': feature_names,
        'target_names': target_names,
        'class_info': CLASS_INFO,
        'feature_info': FEATURE_INFO,
        'n_features': len(feature_names),
        'n_classes': len(target_names),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'hyperparameters': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        },
        'feature_importance': feature_importance,
        'trained_at': datetime.now().isoformat()
    }
    
    metadata_path = Path(model_dir) / 'iris_classifier_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 메타데이터 저장: {metadata_path}")
    
    return model, accuracy, metadata


if __name__ == '__main__':
    model, accuracy, metadata = train_model()
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"정확도: {accuracy:.2%}")
