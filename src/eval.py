import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            preds = (preds > 0.5).int().squeeze()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(" Evaluation for Teacher Model")
    print(f" Accuracy: {acc*100:.2f}%")
    print(f" F1-score: {f1:.4f}")
    print(" Confusion Matrix:\n", cm)
    