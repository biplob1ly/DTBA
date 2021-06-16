import torch
import logging
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from run_manager import RunManager
from torch.utils.data import DataLoader


def train(model, cv_train_datasets, cv_dev_datasets, test_dataset, hyper_params, batch_size, device):
    train_loader = DataLoader(cv_train_datasets[0], batch_size=batch_size, shuffle=True, num_workers=1)
    m = RunManager()
    optimizer = optim.Adam(model.parameters(), lr=hyper_params.learning_rate)

    logging.info("Training Started...")
    m.begin_run(hyper_params, model, train_loader)
    for epoch in range(hyper_params.num_epoch):
        m.begin_epoch(epoch + 1)
        model.train()
        for batch in train_loader:
            drugs = batch['drug'].to(device)
            proteins = batch['protein'].to(device)
            true_affins = batch['affinity'].to(device)
            pred_affins = model(drugs, proteins)
            loss = F.mse_loss(pred_affins, true_affins)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m.track_loss(loss)
            # m.track_num_correct(pred_affins, true_affins)

        m.end_epoch()
    m.end_run()
    hype = f'lr_{hyper_params.learning_rate}_epoch_{hyper_params.num_epoch}'
    m.save(f'./results/train_results_{hype}')
    logging.info("Training finished.\n")

    # Training
    dtset = 'train'
    train_loader = DataLoader(cv_train_datasets[0], batch_size=batch_size, shuffle=True, num_workers=1)
    pred_affins, true_affins, MSE = evaluate(model, train_loader, device)
    compute_scores(pred_affins, true_affins, MSE, dtset, hype)

    # Validation
    dtset = 'dev'
    dev_loader = DataLoader(cv_dev_datasets[0], batch_size=batch_size, shuffle=True, num_workers=1)
    pred_affins, true_affins, MSE = evaluate(model, dev_loader, device)
    compute_scores(pred_affins, true_affins, MSE, dtset, hype)

    # test_dataset
    dtset = 'test'
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    pred_affins, true_affins, MSE = evaluate(model, test_loader, device)
    compute_scores(pred_affins, true_affins, MSE, dtset, hype)


def evaluate(model, loader, device):
    total_loss = 0
    total_batch = 0
    all_pred_affins = torch.tensor([])
    all_true_affins = torch.tensor([])

    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        for batch in loader:
            drugs = batch['drug'].to(device)
            proteins = batch['protein'].to(device)
            true_affins = batch['affinity']

            pred_affins = model(drugs, proteins)
            pred_affins = pred_affins.detach().cpu()

            all_pred_affins = torch.cat((all_pred_affins, pred_affins), dim=0)
            all_true_affins = torch.cat((all_true_affins, true_affins), dim=0)
            loss = F.mse_loss(pred_affins, true_affins)
            total_loss += loss.item()
            total_batch += 1

    all_pred_affins = all_pred_affins.view(-1).tolist()
    all_true_affins = all_true_affins.view(-1).tolist()
    MSE = total_loss / total_batch
    return all_pred_affins, all_true_affins, MSE


def save_predictions(pred_affins, true_affins, dtset, hype):
    pred_affins = np.array(pred_affins)
    true_affins = np.array(true_affins)
    np.savetxt(f'./results/{dtset}_pred_affins_{hype}.txt', pred_affins)
    np.savetxt(f'./results/{dtset}_true_affins_{hype}.txt', true_affins)


def get_cindex(pred_affins, true_affins):
    score = 0
    pair = 0
    for i in range(1, len(true_affins)):
        for j in range(0, i):
            if i is not j:
                if true_affins[i] > true_affins[j]:
                    pair += 1
                    score += 1 * (pred_affins[i] > pred_affins[j]) + 0.5 * (pred_affins[i] == pred_affins[j])
    if pair is not 0:
        return score / pair
    else:
        return 0


def compute_scores(pred_affins, true_affins, MSE, dtset, hype):
    save_predictions(pred_affins, true_affins, dtset, hype)
    logging.info(f"{dtset} MSE: {MSE}")
    CI = get_cindex(pred_affins, true_affins)
    logging.info(f"{dtset} CI: {CI}\n")
    print(f"\n{dtset} MSE: {MSE}"
          f"\n{dtset} CI: {CI}")
