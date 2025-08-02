import cv2                                  #pip install opencv-python
import torch                                #pip install torch
import torch.nn as nn                       
import torch.optim as optim
import numpy as np                          #pip install numpy
import matplotlib.pyplot as plt             #pip install matplotlib
import random
from sklearn.metrics import roc_curve, auc  #pip install scikit-learn

#轉灰階
def r_t_g(r, g, b):
    color = np.uint8([[[r, g, b]]])
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    return gray[0][0] / 255.0

#資料與標籤
def g_d(n=300):
    data = []
    for _ in range(n):
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        gray = r_t_g(r, g, b)
        label = 1 if gray > 0.5 else 0    # 初始標籤
        data.append((gray, label))
    return data

def t(data):
    X = torch.tensor([[x] for x, _ in data], dtype=torch.float32)
    y = torch.tensor([[y] for _, y in data], dtype=torch.float32)
    return X, y

class a(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)  # 回傳 logit，不做 sigmoid

if __name__ == "__main__":
    # 資料
    data = g_d(300)
    X, y = t(data)

    # 模型、優化器、損失
    model = a()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    # 訓練
    for epoch in range(1000):
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}  Loss: {loss.item():.4f}")

    #決策邊界
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    auto_thr = -b / w
    print(f"\nAutomatic decision boundaries (logit=0): x = {auto_thr:.3f}")

    #計算所有機率、ROC 曲線與最佳閾值
    with torch.no_grad():
        logits_all = model(X).numpy().flatten()
        probs_all  = torch.sigmoid(model(X)).numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y.numpy().flatten(), probs_all)
    roc_auc = auc(fpr, tpr)
    best_idx = np.argmax(tpr - fpr)
    roc_thr  = thresholds[best_idx]
    print(f"ROC AUC: {roc_auc:.3f}，optimal threshold (Youden’s J): {roc_thr:.3f}")

    #資料點 + sigmoid 曲線 + auto_thr + roc_thr
    x_line = np.linspace(0,1,200).reshape(-1,1).astype(np.float32)
    with torch.no_grad():
        y_line = torch.sigmoid(model(torch.tensor(x_line))).numpy()

    gray_vals = [x for x,_ in data]
    labels    = [y for _,y in data]

    plt.figure(figsize=(10,4))

    #散點與曲線
    plt.subplot(1,2,1)
    plt.scatter(gray_vals, labels, alpha=0.6, label='data point')
    plt.plot(x_line, y_line, 'r-', label='sigmoid prediction curve')
    plt.axvline(auto_thr, color='g', linestyle='--', label=f'automatic boundary x={auto_thr:.2f}')
    plt.axvline(roc_thr , color='m', linestyle=':',  label=f'ROC optimal threshold x={roc_thr:.2f}')
    plt.xlabel('Grayscale value')
    plt.ylabel('Tags / Prediction Probability')
    plt.title('Perceptron grayscale classification')
    plt.legend()
    plt.grid(True)

    #ROC 曲線
    plt.subplot(1,2,2)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red',
                label=f'sweet spot\n(thr={roc_thr:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
