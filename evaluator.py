import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, jaccard_score

def evaluate(model, dataloader, device, threshold=0.6):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            # flatten để tính metric sklearn
            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_labels.append(masks.cpu().numpy().reshape(-1))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    miou = jaccard_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"Overall Accuracy:   {acc:.4f}")
    print(f"Precision:          {prec:.4f}")
    print(f"Recall:             {rec:.4f}")
    print(f"F1-Score:           {f1:.4f}")
    print(f"Kappa Coefficient:  {kappa:.4f}")
    print(f"Mean IoU:           {miou:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'kappa': kappa,
        'miou': miou,
        'confusion_matrix': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
    }
