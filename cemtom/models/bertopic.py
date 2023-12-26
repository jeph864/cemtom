import bertopic


class Bertopic(bertopic.BERTopic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
