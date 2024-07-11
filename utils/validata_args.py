from utils.exceptions import InvalidTaskException, InvalidEmbeddingMethodException, \
    InvalidClassificationMethodException, InvalidIAAIndexException


def validate_task(args):
    valid_tasks = ['Propaganda', 'Bias']

    match args.task:
        case args.task if args.task in valid_tasks:
            task = args.task
        case _:
            raise InvalidTaskException(args.task)

    return task


def validate_embedding_method(args):
    valid_embedding_methods = ['ML-E5-large', 'BGE-M3', 'E5-mistral-7b', 'Nomic-Embed']

    match args.embedding_method:
        case args.embedding_method if args.embedding_method in valid_embedding_methods:
            embedding_method = args.embedding_method
        case _:
            raise InvalidEmbeddingMethodException(args.embedding_method)

    return embedding_method


def validate_cls_methods(args):
    valid_cls_methods = ["knn", "svc", "xgboost"]

    match args.cls_method:
        case args.cls_method if args.cls_method in valid_cls_methods:
            cls_method = args.cls_method
        case _:
            raise InvalidClassificationMethodException(args.cls_method)

    return cls_method


def validate_iaa_index(args):
    valid_iaa_indices = [1, 2, 3, 4]

    match args.iaa_index:
        case args.iaa_index if args.iaa_index in valid_iaa_indices:
            iaa_index = args.iaa_index
        case _:
            raise InvalidIAAIndexException(args.iaa_index)

    return iaa_index


def validate_args(args):
    task = None
    try:
        task = validate_task(args)
    except InvalidTaskException(args.task) as invalid_task_exception:
        print(invalid_task_exception)

    embedding_method = None
    try:
        embedding_method = validate_embedding_method(args)
    except InvalidEmbeddingMethodException(args.embedding_method) as invalid_embedding_method_exception:
        print(invalid_embedding_method_exception)

    cls_method = None
    try:
        cls_method = validate_cls_methods(args)
    except InvalidClassificationMethodException(args.cls_method) as invalid_cls_method_exception:
        print(invalid_cls_method_exception)

    args = {
        "task": task,
        "embedding_method": embedding_method,
        "cls_method": cls_method
    }

    return args
