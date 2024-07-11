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


class InvalidClassificationMethodException(Exception):
    def __init__(self, cls_method, message="Classification method {0} is not valid."):
        self.message = message
        self.cls_method = cls_method
        super().__init__(self.message.format(self.cls_method))


class InvalidIAAIndexException(Exception):
    def __init__(self, iaa_index, message="IAA index {0} is not valid."):
        self.message = message
        self.iaa_index = iaa_index
        super().__init__(self.message.format(self.iaa_index))
