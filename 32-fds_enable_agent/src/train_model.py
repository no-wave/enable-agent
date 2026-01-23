"""
이상치 탐지 모델 학습 모듈

Isolation Forest와 Autoencoder를 앙상블하여 사기 거래를 탐지한다.
정상 거래 데이터만 사용하여 학습하는 비지도 학습 방식이다.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve


class FraudAutoencoder(nn.Module):
    """이상치 탐지용 Autoencoder"""
    
    def __init__(self, input_dim: int = 10, latent_dim: int = 4):
        super(FraudAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error


def load_and_preprocess_data(data_path: str) -> tuple:
    """데이터를 로드하고 전처리한다."""
    print("데이터 로드 중...")
    df = pd.read_csv(data_path)
    print(f"데이터 로드 완료: {df.shape[0]:,}건")
    
    # 시간 특성 생성
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    df['time_diff_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    df['signup_hour'] = df['signup_time'].dt.hour
    df['purchase_hour'] = df['purchase_time'].dt.hour
    df['is_weekend'] = df['purchase_time'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_night'] = df['purchase_hour'].apply(lambda x: 1 if 0 <= x < 6 else 0)
    
    # 범주형 인코딩
    label_encoders = {}
    for col in ['source', 'browser', 'sex']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # 특성 선택
    feature_names = [
        'purchase_value', 'age', 'time_diff_hours',
        'signup_hour', 'purchase_hour', 'is_weekend', 'is_night',
        'source_encoded', 'browser_encoded', 'sex_encoded'
    ]
    
    X = df[feature_names].values
    y = df['class'].values
    
    fraud_rate = y.mean() * 100
    print(f"사기 비율: {fraud_rate:.2f}%")
    
    return X, y, feature_names, label_encoders


def train_model(
    data_path: str,
    model_dir: str = 'models',
    num_epochs: int = 50,
    batch_size: int = 256
):
    """이상치 탐지 모델을 학습하고 저장한다."""
    
    print("=" * 60)
    print("FDS 이상치 탐지 모델 학습")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # 데이터 로드
    X, y, feature_names, label_encoders = load_and_preprocess_data(data_path)
    
    # 데이터 분할
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_full, X_val, y_train_full, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    
    # 정상 데이터만 추출 (이상치 탐지 핵심)
    normal_mask = y_train_full == 0
    X_train_normal = X_train_full[normal_mask]
    
    print(f"\n학습 셋 (정상만): {X_train_normal.shape[0]:,}개")
    print(f"검증 셋: {X_val.shape[0]:,}개")
    print(f"테스트 셋: {X_test.shape[0]:,}개")
    
    # 정규화
    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ========== Isolation Forest ==========
    print("\n[1/2] Isolation Forest 학습 중...")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_normal_scaled)
    print("Isolation Forest 학습 완료")
    
    # ========== Autoencoder ==========
    print("\n[2/2] Autoencoder 학습 중...")
    
    autoencoder = FraudAutoencoder(input_dim=10, latent_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    train_tensor = torch.FloatTensor(X_train_normal_scaled)
    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss = 0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch [{epoch+1:3d}/{num_epochs}] Loss: {avg_loss:.6f}")
    
    print("Autoencoder 학습 완료")
    
    # Autoencoder 임계값 설정
    autoencoder.eval()
    train_tensor_gpu = torch.FloatTensor(X_train_normal_scaled).to(device)
    train_recon_errors = autoencoder.get_reconstruction_error(train_tensor_gpu).cpu().numpy()
    ae_threshold = np.percentile(train_recon_errors, 95)
    
    # ========== 앙상블 평가 ==========
    print("\n앙상블 모델 평가 중...")
    
    def normalize_scores(scores):
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    
    # 검증 셋 점수 계산
    val_iso_scores = -iso_forest.score_samples(X_val_scaled)
    val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    val_ae_scores = autoencoder.get_reconstruction_error(val_tensor).cpu().numpy()
    
    val_iso_norm = normalize_scores(val_iso_scores)
    val_ae_norm = normalize_scores(val_ae_scores)
    val_ensemble_scores = (val_iso_norm + val_ae_norm) / 2
    
    # 최적 임계값 찾기
    precision, recall, thresholds = precision_recall_curve(y_val, val_ensemble_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    ensemble_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # 테스트 셋 평가
    test_iso_scores = -iso_forest.score_samples(X_test_scaled)
    test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    test_ae_scores = autoencoder.get_reconstruction_error(test_tensor).cpu().numpy()
    
    test_iso_norm = normalize_scores(test_iso_scores)
    test_ae_norm = normalize_scores(test_ae_scores)
    test_ensemble_scores = (test_iso_norm + test_ae_norm) / 2
    
    test_pred = (test_ensemble_scores >= ensemble_threshold).astype(int)
    test_auc = roc_auc_score(y_test, test_ensemble_scores)
    
    print(f"\n테스트 결과:")
    print(f"  AUC-ROC: {test_auc:.4f}")
    print(f"  앙상블 임계값: {ensemble_threshold:.4f}")
    print(f"\n{classification_report(y_test, test_pred, target_names=['Normal', 'Fraud'])}")
    
    # ========== 모델 저장 ==========
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(iso_forest, model_dir / 'isolation_forest.pkl')
    print(f"\nIsolation Forest 저장: {model_dir / 'isolation_forest.pkl'}")
    
    torch.save({
        'model_state_dict': autoencoder.state_dict(),
        'input_dim': 10,
        'latent_dim': 4,
        'threshold': float(ae_threshold)
    }, model_dir / 'autoencoder.pth')
    print(f"Autoencoder 저장: {model_dir / 'autoencoder.pth'}")
    
    joblib.dump(scaler, model_dir / 'fds_scaler.pkl')
    print(f"스케일러 저장: {model_dir / 'fds_scaler.pkl'}")
    
    joblib.dump(label_encoders, model_dir / 'label_encoders.pkl')
    print(f"레이블 인코더 저장: {model_dir / 'label_encoders.pkl'}")
    
    # 메타데이터
    metadata = {
        'model_type': 'Ensemble Anomaly Detection (Isolation Forest + Autoencoder)',
        'feature_names': feature_names,
        'categorical_features': ['source', 'browser', 'sex'],
        'encoding_maps': {
            col: dict(zip(le.classes_.tolist(), range(len(le.classes_))))
            for col, le in label_encoders.items()
        },
        'ensemble_threshold': float(ensemble_threshold),
        'autoencoder_threshold': float(ae_threshold),
        'test_auc_roc': float(test_auc),
        'training_samples': int(X_train_normal.shape[0]),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'trained_at': datetime.now().isoformat()
    }
    
    with open(model_dir / 'fds_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"메타데이터 저장: {model_dir / 'fds_metadata.json'}")
    
    print("\n" + "=" * 60)
    print("모델 학습 완료")
    print("=" * 60)
    
    return iso_forest, autoencoder, scaler, metadata


if __name__ == '__main__':
    train_model('Fraud_Data.csv')
