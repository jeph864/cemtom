class Pipeline:
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass


class BaseParams:
    pass


class BaseHyperparameters(BaseParams):
    pass


class TopicModel:
    pass


class NTMBase(TopicModel):
    pass


class CEModelBase(TopicModel):
    pass


class TrainerBase(BaseParams):
    pass


class EvaluationBase(BaseParams):
    pass
