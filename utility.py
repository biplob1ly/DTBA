import json
import math
import pickle
import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from copy import deepcopy
from collections import OrderedDict
from constants import DRUG_CHARSET, TARGET_CHARSET


class DTIDataset(Dataset):
    def __init__(self, drugs, targets, affinities):
        self.drugs = drugs
        self.targets = targets
        self.affinities = affinities

    def __len__(self):
        return len(self.affinities)

    def __getitem__(self, index):
        drug = self.drugs[index]
        target = self.targets[index]
        affinity = self.affinities[index]
        return drug, target, affinity


def one_hot_smiles(line, max_drug_len):
    X = np.zeros((max_drug_len, len(DRUG_CHARSET)))

    for i, ch in enumerate(line[:max_drug_len]):
        X[i, DRUG_CHARSET[ch] - 1] = 1

    return X


def one_hot_sequence(line, max_target_len):
    X = np.zeros((max_target_len, len(TARGET_CHARSET)))
    for i, ch in enumerate(line[:max_target_len]):
        X[i, TARGET_CHARSET[ch] - 1] = 1

    return X


def label_smiles(line, max_drug_len):
    X = np.zeros(max_drug_len)
    for i, ch in enumerate(line[:max_drug_len]):
        X[i] = DRUG_CHARSET[ch]

    return X


def label_sequence(line, max_target_len):
    X = np.zeros(max_target_len)

    for i, ch in enumerate(line[:max_target_len]):
        X[i] = TARGET_CHARSET[ch]

    return X


def load_dataset(args):
    train_folds = json.load(open(args.dataset_path + "folds/train_fold_setting" + str(args.data_config_id)+".txt"))
    test_set = json.load(open(args.dataset_path + "folds/test_fold_setting" + str(args.data_config_id) + ".txt"))
    return train_folds, test_set


def parse_data(args, with_label=True):
    fpath = args.dataset_path

    drugs = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    targets = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)

    Y = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    if args.is_log:
        Y = -(np.log10(Y / (math.pow(10, 9))))

    XD = []
    XT = []

    if with_label:
        for d in drugs.keys():
            XD.append(label_smiles(drugs[d], args.max_drug_len))

        for t in targets.keys():
            XT.append(label_sequence(targets[t], args.max_target_len))
    else:
        for d in drugs.keys():
            XD.append(one_hot_smiles(drugs[d], args.max_drug_len))

        for t in targets.keys():
            XT.append(one_hot_sequence(targets[t], args.max_target_len))

    return XD, XT, Y


def pack_dataset(XD, XT, Y, rows, cols):
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
    affinities = np.array(affinities).reshape(-1, 1)
    dti_dataset = DTIDataset(drug_data[:256], target_data[:256], affinities[:256])  # Remove [:256], added just to work on a subset
    return dti_dataset


def process_dataset(args):
    logging.info("Loading Data Set...")

    args.drug_charset = DRUG_CHARSET
    args.drug_charset_size = len(DRUG_CHARSET)
    args.target_charset = TARGET_CHARSET
    args.target_charset_size = len(TARGET_CHARSET)

    # Get vectorize form of all drugs, targets and affinities between all of the pairs
    XD, XT, Y = parse_data(args)
    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)
    args.drug_count = XD.shape[0]
    args.target_count = XT.shape[0]

    valid_drug_idxs, valid_target_idxs = np.where(np.isnan(Y) == False)

    train_folds, test_set = load_dataset(args)
    train_sets = []
    dev_sets = []
    for fold_idx in range(len(train_folds)):
        dev_set = train_folds[fold_idx]
        dev_sets.append(dev_set)
        rest_train_folds = deepcopy(train_folds)
        rest_train_folds.pop(fold_idx)
        train_set = [item for sublist in rest_train_folds for item in sublist]
        train_sets.append(train_set)
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
            dti_train_dataset = pack_dataset(XD, XT, Y, train_drug_idxs, train_target_idxs)
            cv_train_datasets.append(dti_train_dataset)

            dev_meta_idxs = dev_sets[fold_idx]
            dev_drug_idxs = valid_drug_idxs[dev_meta_idxs]
            dev_target_idxs = valid_target_idxs[dev_meta_idxs]
            dti_dev_dataset = pack_dataset(XD, XT, Y, dev_drug_idxs, dev_target_idxs)
            cv_dev_datasets.append(dti_dev_dataset)

    test_meta_idxs = test_set
    test_drug_idxs = valid_drug_idxs[test_meta_idxs]
    test_target_idxs = valid_target_idxs[test_meta_idxs]
    dti_test_dataset = pack_dataset(XD, XT, Y, test_drug_idxs, test_target_idxs)

    return cv_train_datasets, cv_dev_datasets, dti_test_dataset
