import argparse
import os
from pathlib import Path

import pandas as pd

from data.sampling_spec import get_sampling_spec
from utils.validata_args import validate_args


def sample_from_label(data: pd.DataFrame, task: str, label_name: str, num_sample: int):
    label_data = data[data[f'{task}'] == label_name]
    sample_data = label_data.sample(n=num_sample, random_state=1)
    return sample_data


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns=['Unnamed: 0', 'Comments', 'Annotator ID'])
    assert data.isna().any().sum() == 0
    data = data.drop_duplicates()
    data_copy = data.copy()

    data_copy["text_tokens_count"] = data_copy["Text"].str.replace(',', '').str.split().str.len()
    data_copy["text_length"] = data_copy["Text"].str.len()

    return data_copy


def prepare_train_test_data(data: pd.DataFrame, label_samples: dict, task: str) -> dict:
    test_data = pd.DataFrame()
    train_data = pd.DataFrame()

    for label, sample_size in label_samples.items():
        sample_data = sample_from_label(data, task, label, sample_size)
        remain_data = pd.concat(
            [data[data[f'{task}'] == label], sample_data]
        ).drop_duplicates(keep=False)

        train_data = pd.concat([train_data, remain_data])
        test_data = pd.concat([test_data, sample_data])

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    assert train_data.shape[0] + test_data.shape[0] == data.shape[0]
    data_splits = {"train": train_data, "test": test_data}

    return data_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    args = validate_args(args)

    task = args["task"]
    label_samples = get_sampling_spec()

    data_dir = os.path.join(os.getcwd(), 'data')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    subset_data = pd.read_excel(os.path.join(data_dir, 'annotated_data.xlsx'))
    preprocessed_data = preprocess(subset_data)
    task_data_splits = prepare_train_test_data(preprocessed_data, label_samples[task], task)

    task_data_dir = os.path.join(data_dir, task.lower())
    Path(task_data_dir).mkdir(parents=True, exist_ok=True)

    task_data_splits["train"].to_excel(
        os.path.join(task_data_dir, f"{task.lower()}_train_data.xlsx"),
        index=False
    )
    task_data_splits["test"].to_excel(
        os.path.join(task_data_dir, f"{task.lower()}_test_data.xlsx"),
        index=False
    )


if __name__ == '__main__':
    main()
