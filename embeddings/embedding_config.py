import torch


def get_embedding_model_spec():
    embedding_model_spec = {
        'ML-E5-large': {
            'model_name': 'intfloat/multilingual-e5-large',
            'max_length': 512,
            'pooling_type': 'mean',
            'normalize': True,
            'batch_size': 1,
            'kwargs': {'device_map': 'cuda', 'torch_dtype': torch.float16}
        },
        'BGE-M3': {
            'model_name': 'BAAI/bge-m3',
            'max_length': 8192,
            'pooling_type': 'cls',
            'normalize': True,
            'batch_size': 1,
            'kwargs': {'device_map': 'cuda', 'torch_dtype': torch.float16}
        },
        'E5-mistral-7b': {
            'model_name': 'intfloat/e5-mistral-7b-instruct',
            'max_length': 32768,
            'pooling_type': 'last_token',
            'normalize': True,
            'batch_size': 1,
            'kwargs': {'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.float16}
        },
        'Nomic-Embed': {
            'model_name': 'nomic-ai/nomic-embed-text-v1',
            'max_length': 8192,
            'pooling_type': 'mean',
            'normalize': True,
            'batch_size': 1,
            'kwargs': {'device_map': 'cuda', 'trust_remote_code': True}
        }
    }

    return embedding_model_spec
