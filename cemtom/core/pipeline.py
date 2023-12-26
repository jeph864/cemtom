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


class AbstractModel:
    pass


class NTMBase(AbstractModel):
    pass


class CEModelBase(AbstractModel):
    pass


class TrainerBase(BaseParams):
    pass


class EvaluationBase(BaseParams):
    pass
