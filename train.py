import torch
from torch.utils.data import DataLoader
from models.anomaly_net import AnomalyNet
from utils.dataset import UCFCrimeDataset

def mil_ranking_loss(a_scores, n_scores, lambda1=8e-5, lambda2=8e-5):
    a_max = torch.max(a_scores, dim=1)[0]
    n_max = torch.max(n_scores, dim=1)[0]
    hinge   = torch.mean(torch.clamp(1.0 - a_max + n_max, min=0.0))
    smooth  = torch.mean((a_scores[:, :-1] - a_scores[:, 1:]) ** 2)
    sparsity = torch.mean(a_scores)
    return hinge + lambda1 * smooth + lambda2 * sparsity

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on: {device}")

    dataset = UCFCrimeDataset(
        anomaly_dir='data/features/train/anomaly',
        normal_dir='data/features/train/normal',
        n_segments=32
    )
    loader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=0)

    model = AnomalyNet(input_dim=1024).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for anomaly_feats, normal_feats in loader:
            anomaly_feats = anomaly_feats.to(device)
            normal_feats  = normal_feats.to(device)
            optimizer.zero_grad()
            a_scores = model(anomaly_feats)
            n_scores = model(normal_feats)
            loss = mil_ranking_loss(a_scores, n_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:02d}/50  |  Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), 'anomaly_net_weights.pth')
    print("Model saved.")

if __name__ == '__main__':
    train()
