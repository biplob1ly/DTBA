import logging
import constants
from utility import *
from copy import deepcopy
from models import *
import torch


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    affinities = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinities.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinities


def prepare_dataset():
    logging.info("Loading Data Set...")

    args.drug_charset = constants.DRUG_CHARSET
    args.drug_charset_size = len(constants.DRUG_CHARSET)
    args.target_charset = constants.TARGET_CHARSET
    args.target_charset_size = len(constants.TARGET_CHARSET)

    data_set = DataSet()
    # Get vectorize form of all drugs, targets and affinities between all of the pairs
    XD, XT, Y = data_set.parse_data(args)
    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)
    args.drug_count = XD.shape[0]
    args.target_count = XT.shape[0]

    valid_drug_idxs, valid_target_idxs = np.where(np.isnan(Y) == False)

    train_folds, test_set = data_set.load_dataset(args)
    train_sets = []
    dev_sets = []
    test_sets = []
    for fold_idx in range(len(train_folds)):
        dev_set = train_folds[fold_idx]
        dev_sets.append(dev_set)
        rest_train_folds = deepcopy(train_folds)
        rest_train_folds.pop(fold_idx)
        train_set = [item for sublist in rest_train_folds for item in sublist]
        train_sets.append(train_set)
        test_sets.append(test_set)
        logging.info(f"CV set {fold_idx+1}"
                     f"\nTrain set size: {len(train_set)}"
                     f"\nDev set size: {len(dev_set)}"
                     f"\nTest set size: {len(test_set)}\n\n")

    cv_train_datasets = []
    cv_dev_datasets = []
    for fold_idx in range(len(dev_sets)):
        if fold_idx == 0:
            train_meta_idxs = train_sets[fold_idx]
            train_drug_idxs = valid_drug_idxs[train_meta_idxs]
            train_target_idxs = valid_target_idxs[train_meta_idxs]
            train_drugs, train_targets, train_affinities = prepare_interaction_pairs(XD, XT, Y, train_drug_idxs, train_target_idxs)
            cv_train_datasets.append((train_drugs, train_targets, train_affinities))

            dev_meta_idxs = dev_sets[fold_idx]
            dev_drug_idxs = valid_drug_idxs[dev_meta_idxs]
            dev_target_idxs = valid_target_idxs[dev_meta_idxs]
            dev_drugs, dev_targets, dev_affinities = prepare_interaction_pairs(XD, XT, Y, dev_drug_idxs, dev_target_idxs)
            cv_dev_datasets.append((dev_drugs, dev_targets, dev_affinities))

    return cv_train_datasets, cv_dev_datasets


def run():
    cv_train_datasets, cv_dev_datasets = prepare_dataset()
    train_drugs, train_targets, train_affinities = cv_train_datasets[0]
    dev_drugs, dev_targets, dev_affinities = cv_dev_datasets[0]

    # One can use train_drugs[78836 x 100], train_targets[78836 x 1000], train_affinities for training
    print("Drug original shape: ", train_drugs.shape)
    print("Target original shape: ", train_targets.shape)

    # CNNModel is just a baseline model, one can build better model based on the prepared data set
    cnn_model = CNNModel(args)
    outs = cnn_model(torch.from_numpy(train_drugs[:2]), torch.from_numpy(train_targets[:2]))
    print("No training, 2 drug-target pair affinity prediction:\n", outs)


if __name__ == "__main__":
    args = constants.get_args()
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='app.log', filemode='w', format=FORMAT, level=getattr(logging, args.log.upper()))
    logging.info(args)
    run()
