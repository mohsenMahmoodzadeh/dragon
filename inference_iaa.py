import argparse
import os
from pathlib import Path

import pandas as pd

import torch

from classification_config import create_cls_model_spec
from data.data_spec import get_data_spec
from embeddings.embedding_config import get_embedding_model_spec
from generate_embeddings import create_embeddings
from get_train_test_splits import get_train_test_splits
from utils.validata_args import validate_args


def get_iaa_data(iaa_data_dir, iaa_index):
    iaa_data = pd.read_excel(os.path.join(iaa_data_dir, f'iaa{iaa_index}.xlsx'))
    return iaa_data


def get_train_data(task, data_spec):
    data_splits = get_train_test_splits(task, data_spec)
    train_data = data_splits["train"]

    return train_data


def generate_embeddings(texts, embedding_model_spec, embedding_method, path):
    create_embeddings(embedding_method, embedding_model_spec, texts, path)
    embeddings = torch.load(path)
    return embeddings


def predict(cls, train_data, x_pred, id2label):
    x_train, y_train = train_data
    cls.fit(x_train, y_train)
    y_pred = cls.predict(x_pred)
    predictions = [id2label[str(y_pred)] for y_pred in y_pred]
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--embedding_method", type=str)
    parser.add_argument("--cls_method", type=str)
    parser.add_argument("--iaa_index", type=int)
    args = parser.parse_args()
    args = validate_args(args)

    task = args["task"]
    embedding_method = args["embedding_method"]
    iaa_index = str(args["iaa_index"])

    data_spec = get_data_spec()
    data_dir = os.path.join(os.getcwd(), 'data')
    train_data = get_train_data(task, data_spec)
    train_texts = train_data["Text"].tolist()

    iaa_data_dir = os.path.join(data_dir, 'iaa')
    iaa_data = get_iaa_data(iaa_data_dir, iaa_index)
    iaa_texts = iaa_data["Text"].tolist()

    embedding_dir = os.path.join(os.getcwd(), 'embeddings')
    Path(os.path.join(embedding_dir, 'iaa', task.lower())).mkdir(parents=True, exist_ok=True)
    embedding_model_spec = get_embedding_model_spec()

    train_embeddings = generate_embeddings(
        train_texts,
        embedding_model_spec,
        embedding_method,
        os.path.join(embedding_dir, task.lower(), f'{embedding_method}_train_embeddings.pt')
    )

    iaa_embeddings = generate_embeddings(
        iaa_texts,
        embedding_model_spec,
        embedding_method,
        os.path.join(embedding_dir, 'iaa', task.lower(), f'{embedding_method}_{iaa_index}_embeddings.pt')
    )

    cls_model_spec = create_cls_model_spec(task, data_spec)
    cls = cls_model_spec[args["cls_method"]]["cls"]

    x_train = train_embeddings
    y_train = train_data[args["task"]].tolist()
    x_pred = iaa_embeddings

    task_predictions = predict(cls, (x_train, y_train), x_pred, data_spec[task]['id2label'])
    iaa_data[task] = task_predictions
    iaa_data.to_excel(
        os.path.join(
            data_dir,
            'iaa',
            f"{task.lower()}_{embedding_method}_{args['cls_method']}_iaa{iaa_index}_data.xlsx"),
        index=False
    )


if __name__ == '__main__':
    main()
