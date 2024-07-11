import argparse
import os
from pprint import pprint

from sklearn.metrics import classification_report
import torch

from classification_config import create_cls_model_spec
from data.data_spec import get_data_spec
from get_train_test_splits import get_train_test_splits
from utils.validata_args import validate_args


def classify(cls, train_data, test_data, label_names):
    x_train, y_train = train_data
    x_test, y_test = test_data

    cls.fit(x_train, y_train)
    y_pred = cls.predict(x_test)

    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    return report


def get_embeddings(args):
    train_embeddings = torch.load(
        os.path.join(os.getcwd(), 'embeddings', args["task"].lower(), f'{args["embedding_method"]}_train_embeddings.pt')
    )

    test_embeddings = torch.load(
        os.path.join(os.getcwd(), 'embeddings', args["task"].lower(), f'{args["embedding_method"]}_test_embeddings.pt')
    )

    embeddings = {
        "train": train_embeddings,
        "test": test_embeddings
    }

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--embedding_method", type=str)
    parser.add_argument("--cls_method", type=str)
    args = parser.parse_args()
    args = validate_args(args)

    data_spec = get_data_spec()
    cls_model_spec = create_cls_model_spec(args["task"], data_spec)

    cls = cls_model_spec[args["cls_method"]]["cls"]

    embeddings = get_embeddings(args)
    train_embeddings = embeddings["train"]
    test_embeddings = embeddings["test"]

    data_splits = get_train_test_splits(args["task"], data_spec)
    train_data = data_splits["train"]
    test_data = data_splits["test"]

    x_train = train_embeddings
    y_train = train_data[args["task"]].tolist()
    x_test = test_embeddings
    y_test = test_data[args["task"]].tolist()

    label_names = data_spec[args["task"]]["label_names"]
    report = classify(cls, (x_train, y_train), (x_test, y_test), label_names)

    print(
        f"The classification report for {args['task']} task with {args['embedding_method']} "
        f"embedding method and {args['cls_method']} classifier:\n"
    )

    pprint(report)


if __name__ == '__main__':
    main()
