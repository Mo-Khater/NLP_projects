
def train_model(model,criterian,optim,loader,epochs):
    model.train()
    for epoch in range(epochs):
        for batch_x,batch_y in loader:
            optim.zero_grad()
            pred_probs = model(batch_x)
            loss = criterian(pred_probs,batch_y)
            loss.backward()
            optim.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


def evaluate(model, loader):
    from sklearn.metrics import f1_score
    import torch

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            pred_probs = model(batch_x)
            y_pred_bin = (pred_probs > 0.5).int()

            y_true.append(batch_y)
            y_pred.append(y_pred_bin)

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    print("y_true sample:\n", y_true[:5])
    print("y_pred sample:\n", y_pred[:5])

    f1_micro = f1_score(y_true, y_pred, average='micro')
    return f1_micro

