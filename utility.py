import json
import math
import pickle
from collections import OrderedDict
import numpy as np
from constants import DRUG_CHARSET, TARGET_CHARSET


class DataSet:

    @staticmethod
    def load_dataset(args):
        train_folds = json.load(open(args.dataset_path + "folds/train_fold_setting" + str(args.data_config_id)+".txt"))
        test_set = json.load(open(args.dataset_path + "folds/test_fold_setting" + str(args.data_config_id) + ".txt"))
        return train_folds, test_set

    def parse_data(self, args, with_label=True):
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
    X = np.zeros(max_drug_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_drug_len]):
        X[i] = DRUG_CHARSET[ch]

    return X


def label_sequence(line, max_target_len):
    X = np.zeros(max_target_len, dtype=np.int64)

    for i, ch in enumerate(line[:max_target_len]):
        X[i] = TARGET_CHARSET[ch]

    return X
