"""
PyTorch 딥러닝 모델 학습 모듈

Spaceship Titanic 데이터셋을 사용하여 승객의 다른 차원 이동 여부를 예측하는
신경망 분류 모델을 학습한다.
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


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


def load_data(data_path: str) -> pd.DataFrame:
    """데이터를 로드한다."""
    df = pd.read_csv(data_path)
    print(f"데이터 로드 완료: {df.shape[0]}개 샘플, {df.shape[1]}개 특성")
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.25):
    """데이터를 전처리하고 분할한다."""
    feature_names = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
                     'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    target_name = 'Transported'
    
    X = df[feature_names].values
    y = df[target_name].values
    
    # 학습/테스트 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 학습/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    # 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"학습 셋: {X_train.shape[0]}개")
    print(f"검증 셋: {X_val.shape[0]}개")
    print(f"테스트 셋: {X_test.shape[0]}개")
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'target_name': target_name
    }


def create_dataloaders(data: dict, batch_size: int = 64):
    """PyTorch DataLoader를 생성한다."""
    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.FloatTensor(data['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.FloatTensor(data['y_val'])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(data['X_test']),
        torch.FloatTensor(data['y_test'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """한 에포크 학습을 수행한다."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """모델을 평가한다."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels, all_probs


def train_model(
    data_path: str,
    model_dir: str = 'models',
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    early_stop_patience: int = 15
):
    """모델을 학습하고 저장한다."""
    
    print("=" * 60)
    print("Spaceship Titanic 딥러닝 모델 학습")
    print("=" * 60)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # 데이터 준비
    df = load_data(data_path)
    data = prepare_data(df)
    train_loader, val_loader, test_loader = create_dataloaders(data, batch_size)
    
    # 모델 생성
    model = SpaceshipClassifier(input_dim=10).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 학습
    print(f"\n학습 시작 (에포크: {num_epochs}, Early Stop: {early_stop_patience})")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # 최적 모델 로드
    model.load_state_dict(best_model_state)
    
    # 테스트 평가
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )
    test_auc = roc_auc_score(test_labels, test_probs)
    
    print(f"\n테스트 결과:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC-ROC: {test_auc:.4f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['Not Transported', 'Transported'])}")
    
    # 모델 저장
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'spaceship_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': 10,
        'hidden_dims': [64, 32, 16],
        'dropout_rate': 0.3
    }, model_path)
    print(f"\n모델 저장: {model_path}")
    
    scaler_path = model_dir / 'spaceship_scaler.pkl'
    joblib.dump(data['scaler'], scaler_path)
    print(f"스케일러 저장: {scaler_path}")
    
    metadata = {
        'model_type': 'PyTorch Neural Network',
        'architecture': 'SpaceshipClassifier',
        'input_dim': 10,
        'hidden_dims': [64, 32, 16],
        'dropout_rate': 0.3,
        'feature_names': data['feature_names'],
        'target_name': data['target_name'],
        'target_classes': ['Not Transported', 'Transported'],
        'test_accuracy': float(test_acc),
        'test_auc_roc': float(test_auc),
        'training_samples': int(data['X_train'].shape[0]),
        'scaler_mean': data['scaler'].mean_.tolist(),
        'scaler_scale': data['scaler'].scale_.tolist(),
        'trained_at': datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }
    
    metadata_path = model_dir / 'spaceship_classifier_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"메타데이터 저장: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("모델 학습 완료")
    print("=" * 60)
    
    return model, data['scaler'], metadata


if __name__ == '__main__':
    train_model('spaceship-preprocessing.csv')
