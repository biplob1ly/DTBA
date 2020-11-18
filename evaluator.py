import torch
import torch.nn.functional as F


def get_MSE_CI(model, loader):
    total_loss = 0
    total_samples = 0
    all_preds = torch.tensor([])
    all_affins = torch.tensor([])
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        for batch in loader:
            drugs = batch[0].long()
            targets = batch[1].long()
            affins = batch[2].float()
            preds = model(drugs, targets)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_affins = torch.cat((all_affins, affins), dim=0)
            loss = F.mse_loss(preds, affins)
            total_loss += loss*affins.size(0)
            total_samples += affins.size(0)
    MSE = total_loss / total_samples
    CI = get_cindex(all_preds.view(-1), all_affins.view(-1))
    return MSE, CI


def get_cindex(preds, affins):
    score = 0
    pair = 0
    for i in range(1, len(affins)):
        for j in range(0, i):
            if i is not j:
                if affins[i] > affins[j]:
                    pair += 1
                    score += 1 * (preds[i] > preds[j]) + 0.5 * (preds[i] == preds[j])
    if pair is not 0:
        return score / pair
    else:
        return 0
