from fastai.callbacks import EarlyStoppingCallback
from fastai.tabular import *

from src.classifier.ModelAbstract import ModelAbstract
from src.data.DataCsvInterface import DataCsvInterface


class CustomTabularModel(ModelAbstract):
    def __init__(self, data_split, should_shuffle, n_rows, layers: dict):
        self.input_data = DataCsvInterface().get_data_info(data_split, should_shuffle, n_rows)
        self.model: tabular_learner
        self.reset_params(layers)
        self.args = layers

    def reset_params(self, args):
        """
        So the args might contain layers. This needs to be handled by the model for custom behavior.
        We could add a validation method that parses args for layer related args, and then do something else with other
        params.

        :param args: A dictionary of args to pass to reset the model
        :return:
        """
        self.args = args
        layers = []
        stop_early = True
        dropout = 0.5

        for arg in [_ for _ in args if str(_).__contains__('layer')]:
            layers.append(int(args[arg]))

        for arg in [_ for _ in args if str(_).__contains__('dropout')]:
            dropout = float(args[arg])

        for arg in [_ for _ in args if str(_).__contains__('early_stopping')]:
            stop_early = arg

        if stop_early:
            self.model = tabular_learner(self.input_data, layers=layers, metrics=accuracy, emb_drop=dropout,
                                     callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01,
                                                           patience=3)])
        else:
            self.model = tabular_learner(self.input_data, layers=layers, metrics=accuracy, emb_drop=dropout)

    def train(self, epochs=3, k=1) -> float:
        k_accuracy = []

        for i in range(k):
            # print(f'Starting k: {i}')
            self.model.fit_one_cycle(epochs, 1e-2)
            k_accuracy.append(float(self.model.recorder.metrics[-1][0]))
            if k != 1:
                self.input_data = DataCsvInterface().get_data_info(1 / k, True, 1000)
                self.reset_params(self.args)

        # print(f'Found Accuracies: {k_accuracy}')
        return np.average(k_accuracy)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def test_model(epochs):
        model = CustomTabularModel(0.5, False, 1000, {'layer1': 20, 'layer2': 20, 'dropout':0.5})
        print(model.train(epochs, 10))
