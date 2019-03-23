from abc import abstractmethod, ABC
from fastai.basic_data import DataBunch


class ModelAbstract(ABC):

    @abstractmethod
    def reset_params(self, **kwargs):
        pass

    @abstractmethod
    def train(self, data_bunch: DataBunch) -> float:
        pass

    @abstractmethod
    def predict(self, x):
        pass
