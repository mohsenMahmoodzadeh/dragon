class InvalidTaskException(Exception):
    def __init__(self, task, message="Task {0} is not valid."):
        self.message = message
        self.task = task
        super().__init__(self.message.format(self.task))


class InvalidEmbeddingMethodException(Exception):
    def __init__(self, embedding_method, message="Embedding method {0} is not valid."):
        self.message = message
        self.embedding_method = embedding_method
        super().__init__(self.message.format(self.embedding_method))


class InvalidSplitException(Exception):
    def __init__(self, split, message="Split {0} is not valid."):
        self.message = message
        self.split = split
        super().__init__(self.message.format(self.split))


class InvalidLanguageException(Exception):
    def __init__(self, language, message="Language {0} is not valid."):
        self.message = message
        self.language = language
        super().__init__(self.message.format(self.language))


class InvalidClassificationMethodException(Exception):
    def __init__(self, cls_method, message="Classification method {0} is not valid."):
        self.message = message
        self.cls_method = cls_method
        super().__init__(self.message.format(self.cls_method))
