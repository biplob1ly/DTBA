import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import constants
from utility import *
from models import *
from run_manager import RunManager
from evaluator import *


def get_hyper_params_combinations(args):
    params = OrderedDict(
        num_filters=args.num_filters,
        drug_kernel_size=args.drug_kernel_size,
        target_kernel_size=args.target_kernel_size,
        learning_rate=[args.learning_rate],
        num_epoch=[args.num_epoch]
    )

    HyperParams = namedtuple('HyperParams', params.keys())
    hyper_params_list = []
    for v in product(*params.values()):
        hyper_params_list.append(HyperParams(*v))
    return hyper_params_list


def train(model, loader, hyper_params):
    m = RunManager()
    optimizer = optim.Adam(model.parameters(), lr=hyper_params.learning_rate)

    m.begin_run(hyper_params, model, loader)
    for epoch in range(hyper_params.num_epoch):
        m.begin_epoch(epoch+1)
        for batch in loader:
            drugs = batch[0].long()
            targets = batch[1].long()
            affinities = batch[2].float()
            preds = model(drugs, targets)
            loss = F.mse_loss(preds, affinities)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m.track_loss(loss)
            # m.track_num_correct(preds, affinities)

        m.end_epoch()
    m.end_run()

    m.save('results')
    return model


def run(args):
    ''' cv_train_datasets, cv_dev_datasets each contains 5 DTIDatasets for train and development/validation respectively
    test_dataset is a single DTIDataset.
    Each DTIDataset can be used with dataloader to retrieve drug, target and affinity score as tensor in batch.'''

    cv_train_datasets, cv_dev_datasets, test_dataset = process_dataset(args)
    for hyper_params in get_hyper_params_combinations(args):
        train_loader = DataLoader(cv_train_datasets[0], batch_size=args.batch_size, shuffle=True, num_workers=1)
        model = CNNModel(args.drug_charset_size, args.drug_embedding_dim, hyper_params.drug_kernel_size,
                         args.target_charset_size, args.target_embedding_dim, hyper_params.target_kernel_size,
                         hyper_params.num_filters)
        logging.info(f"Training with: {hyper_params}")
        trained_model = train(model, train_loader, hyper_params)
        logging.info("Training finished")
        dev_loader = DataLoader(cv_dev_datasets[0], batch_size=args.batch_size, shuffle=True, num_workers=1)
        dev_mse, dev_ci = get_MSE_CI(trained_model, dev_loader)
        logging.info(f"Dev MSE: {dev_mse} and Dev CI: {dev_ci}")
        print(f"\nDev MSE: {dev_mse}\nDev CI: {dev_ci}")


if __name__ == "__main__":
    arguments = constants.get_args()
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='app.log', filemode='w', format=FORMAT, level=getattr(logging, arguments.log.upper()))
    logging.info(arguments)
    run(arguments)

