import numpy as np
import torch
import torch.nn as nn

def train_teacher(model, train_loader, epochs=200):
    lr=1e-3
    pos_weight = torch.tensor([len(train_loader) / sum(train_loader)], dtype=torch.float32).to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(device)                        # [B, 3, 3000]
            yb = yb.to(device).unsqueeze(1)           # [B, 1]

            output = model(xb)                        # forward pass
            loss = criterion(output, yb)              # compute loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f" Epoch {epoch}/{epochs} - Loss: {total_loss:.4f}")


def train_student_kd_fixed(student, teacher, train_loader, epochs=200):
    temperature=3.0, alpha=0.9, lr=1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    teacher.to(device)
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    bce_loss = nn.BCELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    eps = 1e-6

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)

            s_out = student(xb)
            s_out = torch.clamp(s_out, eps, 1 - eps)

            with torch.no_grad():
                t_out = teacher(xb)
                t_out = torch.clamp(t_out, eps, 1 - eps)

            loss_hard = bce_loss(s_out, yb)

            log_soft_ratio = torch.log(s_out / t_out)
            loss_soft = kl_loss(log_soft_ratio, t_out)

            loss = alpha * loss_soft + (1 - alpha) * loss_hard

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"KD Epoch {epoch}/{epochs} - Loss: {total_loss:.4f}")

def train_baseline_model()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    for epoch in range(1, 6):
        baseline_model.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            preds = baseline_model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f" Epoch {epoch} - Loss: {total_loss:.4f}")
        
