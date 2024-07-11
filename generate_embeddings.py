import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

from data.data_spec import get_data_spec
from embeddings.embedding_config import get_embedding_model_spec
from get_train_test_splits import get_train_test_splits
from utils.validata_args import validate_args


def mean_pooling(model_output):
    return torch.mean(model_output["last_hidden_state"], dim=1)


def cls_pooling(model_output):
    return model_output[0][:, 0]


def last_token_pooling(model_output):
    return model_output[0][:, -1]


def get_sentence_embedding(
        text,
        tokenizer,
        embed_model,
        normalize,
        max_length,
        pooling_type='cls'
):
    if pooling_type == "last_token":
        encoded_input = tokenizer(
            text,
            max_length=max_length,
            return_attention_mask=False,
            padding=False,
            truncation=True
        )
        encoded_input['input_ids'] = encoded_input['input_ids'] + [tokenizer.eos_token_id]
        encoded_input = tokenizer.pad(
            [encoded_input],
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        ).to("cuda")

    else:
        encoded_input = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to("cuda")

    with torch.no_grad():
        model_output = embed_model(**encoded_input)

    sentence_embeddings = None
    match pooling_type:
        case "cls":
            sentence_embeddings = cls_pooling(model_output)
        case "mean":
            sentence_embeddings = mean_pooling(model_output)
        case "last_token":
            sentence_embeddings = last_token_pooling(model_output)

    if normalize:
        sentence_embeddings = F.normalize(sentence_embeddings)

    return sentence_embeddings


def embed(embed_model, tokenizer, data, model_spec):
    embeddings = [
        get_sentence_embedding(
            sentence,
            tokenizer,
            embed_model,
            model_spec['normalize'],
            model_spec['max_length'],
            model_spec['pooling_type']
        ) for sentence in data
    ]

    embeddings = torch.cat(embeddings)

    return embeddings


def create_embeddings(embedding_method, embedding_model_spec, data, save_to):
    print("Processing model : " + str(embedding_model_spec))

    tokenizer = AutoTokenizer.from_pretrained(
        embedding_model_spec[embedding_method]['model_name']
    )

    embed_model = AutoModel.from_pretrained(
        embedding_model_spec[embedding_method]['model_name'],
        **embedding_model_spec[embedding_method]['kwargs']
    )

    if embedding_method == "Nomic-Embed":
        embed_model.to('cuda')

    embeddings = embed(embed_model, tokenizer, data, embedding_model_spec[embedding_method])
    embeddings = embeddings.detach().cpu()
    print(f"{embedding_method} embedding shape = {embeddings.shape}")
    torch.save(embeddings, save_to)


def generate(data_spec, embedding_model_spec, args):
    task = args["task"]
    embedding_method = args["embedding_method"]

    data_splits = get_train_test_splits(task, data_spec)
    train_data = data_splits["train"]
    test_data = data_splits["test"]

    train_texts = train_data["Text"].tolist()
    test_texts = test_data["Text"].tolist()

    create_embeddings(
        embedding_method,
        embedding_model_spec,
        train_texts,
        os.path.join(os.getcwd(), 'embeddings', task.lower(), f'{embedding_method}_train_embeddings.pt')
    )

    create_embeddings(
        embedding_method,
        embedding_model_spec,
        test_texts,
        os.path.join(os.getcwd(), 'embeddings', task.lower(), f'{embedding_method}_test_embeddings.pt')
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--embedding_method", type=str)
    args = parser.parse_args()
    args = validate_args(args)

    embeddings_dir = os.path.join(os.getcwd(), 'embeddings')
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(embeddings_dir, args["task"].lower())).mkdir(parents=True, exist_ok=True)

    data_spec = get_data_spec()
    embedding_model_spec = get_embedding_model_spec()

    generate(data_spec, embedding_model_spec, args)


if __name__ == '__main__':
    main()
