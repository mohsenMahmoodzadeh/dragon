from utils.exceptions import InvalidTaskException, InvalidEmbeddingMethodException, InvalidSplitException, \
    InvalidLanguageException, InvalidClassificationMethodException


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


def validate_split(args):
    valid_splits = ['train', 'test', 'train_test']

    match args.split:
        case args.split if args.split in valid_splits:
            split = args.split
        case _:
            raise InvalidSplitException(args.split)

    return split


def validate_language(args):
    valid_languages = ["English", "Arabic", "Hebrew", "Hindi", "French", "all"]

    match args.language:
        case args.language if args.language in valid_languages:
            language = args.language
        case _:
            raise InvalidLanguageException(args.language)

    return language


def validate_cls_methods(args):
    valid_cls_methods = ["knn", "svc", "xgboost"]

    match args.cls_method:
        case args.cls_method if args.cls_method in valid_cls_methods:
            cls_method = args.cls_method
        case _:
            raise InvalidClassificationMethodException(args.cls_method)

    return cls_method


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

    split = None
    try:
        split = validate_split(args)
    except InvalidSplitException(args.split) as invalid_split_exception:
        print(invalid_split_exception)

    language = None
    try:
        language = validate_language(args)
    except InvalidLanguageException(args.split) as invalid_language_exception:
        print(invalid_language_exception)

    cls_method = None
    try:
        cls_method = validate_cls_methods(args)
    except InvalidClassificationMethodException(args.cls_method) as invalid_cls_method_exception:
        print(invalid_cls_method_exception)

    args = {
        "task": task,
        "embedding_method": embedding_method,
        "split": split,
        "language": language,
        "cls_method": cls_method
    }

    return args
