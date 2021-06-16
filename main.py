import logging
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import constants
from utility import *
from models import CNNModel, Transformer, TransDTBA
from trainer import train
import random
import os


def get_hyper_params_combinations(args):
    params = OrderedDict(
        num_filters=args.num_filters,
        drug_kernel_size=args.drug_kernel_size,
        protein_kernel_size=args.protein_kernel_size,
        learning_rate=[args.learning_rate],
        num_epoch=[args.num_epoch]
    )

    HyperParams = namedtuple('HyperParams', params.keys())
    hyper_params_list = []
    for v in product(*params.values()):
        hyper_params_list.append(HyperParams(*v))
    return hyper_params_list


def run(args, device):
    ''' cv_train_datasets, cv_dev_datasets each contains 5 DTIDatasets for train and development/validation respectively
    test_dataset is a single DTIDataset.
    Each DTIDataset can be used with dataloader to retrieve drug, protein and affinity score as tensor in batch.'''

    cv_train_datasets, cv_dev_datasets, test_dataset = process_dataset(args)

    for hyper_params in get_hyper_params_combinations(args):
        
        if args.model == 'CNN':
            model = CNNModel(args.drug_charset_size, args.drug_embedding_dim, hyper_params.drug_kernel_size,
                              args.protein_charset_size, args.protein_embedding_dim, hyper_params.protein_kernel_size,
                              hyper_params.num_filters)
            model.to(device)
        elif args.model == 'Transformer':
             model = Transformer(args.drug_charset_size, args.drug_embedding_dim, args.max_drug_len,
                               args.protein_charset_size, args.protein_embedding_dim, args.max_protein_len,
                               args.num_trans_layers, args.num_attn_heads, args.trans_forward_expansion,
                               args.trans_dropout_rate, args.batch_size)
             model.to(device)
        elif args.model == 'TransDTBA':
             model = TransDTBA(args.drug_charset_size, args.drug_embedding_dim, args.max_drug_len,
                               args.protein_charset_size, args.protein_embedding_dim, args.max_protein_len,
                               args.num_trans_layers, args.num_attn_heads, args.trans_forward_expansion,
                               args.trans_dropout_rate, args.batch_size)
             model.to(device)
        else:
            raise ValueError("Unknown value for args.model. Pick CNN or LSTM")

        logging.info(f"Training with: {hyper_params}")
        train(model, cv_train_datasets, cv_dev_datasets, test_dataset, hyper_params, args.batch_size, device)


if __name__ == "__main__":
    arguments = constants.get_args()
    if not os.path.exists('./results'):
        os.makedirs('./results')
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='./results/app.log', filemode='w', format=FORMAT, level=getattr(logging, arguments.log.upper()))
    logging.info(arguments)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    random.seed(arguments.random_seed)
    np.random.seed(arguments.random_seed)
    torch.manual_seed(arguments.random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(arguments.random_seed)
    run(arguments, device)

