import argparse

PADDING_INDEX = 0
KNOWN_DRUG_KNOWN_PROTEIN = 1
KNOWN_DRUG_UNKNOWN_PROTEIN = 2
UNKNOWN_DRUG_KNOWN_PROTEIN = 3

DRUG_CHARSET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

PROTEIN_CHARSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                  "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                  "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                  "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', default="INFO", help="Logging level.")
    parser.add_argument('--random-seed', default=123, help="Random seed.")

    parser.add_argument(
        '--data_config_id',
        type=int,
        default=KNOWN_DRUG_KNOWN_PROTEIN,
        help='Problem Setting (1-4)'
    )

    parser.add_argument(
        '--max_drug_len',
        type=int,
        default=100,
        help='Max Length of SMILES sequence of drug'
    )

    parser.add_argument(
        '--max_protein_len',
        type=int,
        default=1000,
        help='Max Length of FASTA sequence of protein/protein'
    )

    parser.add_argument(
        '--drug_embedding_dim',
        type=int,
        default=128,
        help='Embedding dimension for a drug (SMILES) character'
    )

    parser.add_argument(
        '--protein_embedding_dim',
        type=int,
        default=128,
        help='Embedding dimension for a Target (FASTA) character'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='Transformer',
        help='CNN/Transformer/TransDTBA models'
    )
    
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=128,
        help='Hidden size for LSTM models'
    )
    
    parser.add_argument(
        '--protein_output_size',
        type=int,
        default=128,
        help='Output size of LSTM network for protein (FASTA) sequence'
    )
    
    parser.add_argument(
        '--drug_output_size',
        type=int,
        default=128,
        help='Output size of LSTM network for drug (SMILES) sequence'
    )
    
    parser.add_argument(
        '--drug_kernel_size',
        type=int,
        default=[4],
        nargs='+',
        help='Kernel size for drug (SMILES sequence)'
    )

    parser.add_argument(
        '--protein_kernel_size',
        type=int,
        default=[4],
        nargs='+',
        help='Kernel size for protein (FASTA sequence)'
    )

    parser.add_argument(
        '--num_filters',
        type=int,
        default=[32],
        nargs='+',
        help='Number of filters/out_chanel for initial convolution layer'
    )

    parser.add_argument(
        '--num_trans_layers',
        type=int,
        default=2,
        help='Number of transformer layers'
    )

    parser.add_argument(
        '--num_attn_heads',
        type=int,
        default=8,
        help='Number of transformer attention heads'
    )

    parser.add_argument(
        '--trans_forward_expansion',
        type=int,
        default=4,
        help='Factor to expand transformers hidden representation'
    )

    parser.add_argument(
        '--trans_dropout_rate',
        type=float,
        default=0.1,
        help='Dropout rate for transformers'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )

    parser.add_argument(
        '--num_epoch',
        type=int,
        default=100,
        help='Number of epochs to train.'
    )

    parser.add_argument(
        '--binary_th',
        type=float,
        default=0.0,
        help='Threshold to split data into binary classes'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./data/kiba/',
        help='Directory for input data.'
    )

    parser.add_argument(
        '--is_log',
        type=int,
        default=0,
        help='use log transformation for Y'
    )

    args = parser.parse_args()  # '--protein_kernel_size 4 8'.split()
    return args
