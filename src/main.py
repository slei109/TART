import os
import pickle

from classifier.classifier_getter import get_classifier
from train.train import train
from train.test import test

from tools.tool import parse_args, print_args, set_seed

import dataset.loader as loader
from embedding.embedding import get_embedding


def main():

    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    args.id2word = vocab.itos

    # initialize model
    model = {}
    model["G"] = get_embedding(vocab, args)
    model["clf"] = get_classifier(model["G"].ebd_dim, args)


    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train(train_data, val_data, model, args)


    test_acc, test_std, drawn_data = test(test_data, model, args,
                                          args.test_episodes)


    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
        }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()