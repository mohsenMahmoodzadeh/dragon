import os

import pandas as pd


def get_train_test_splits(task, data_spec):
    task_data_spec = data_spec[task]

    data_dir = os.path.join(os.getcwd(), 'data')
    task_data_dir = os.path.join(data_dir, task.lower())

    train_data = pd.read_excel(os.path.join(task_data_dir, f'{task.lower()}_train_data.xlsx'))
    test_data = pd.read_excel(os.path.join(task_data_dir, f'{task.lower()}_test_data.xlsx'))

    label2id = task_data_spec['label2id']

    train_data[task] = train_data[task].map(label2id)
    test_data[task] = test_data[task].map(label2id)

    data_splits = {"train": train_data, "test": test_data}

    return data_splits
