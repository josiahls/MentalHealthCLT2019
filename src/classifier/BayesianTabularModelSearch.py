import os
from datetime import datetime
from pathlib import Path

from pandas.io.json import json
import pandas as pd
from bayes_opt import BayesianOptimization, Events, JSONLogger
from fastai.basic_data import DataBunch

from src.classifier.CustomTabularModel import CustomTabularModel
from src.data.DataCsvInterface import DataCsvInterface


class BayesianSearcher:
    def __init__(self, epochs):
        self.results = []
        self.epochs = epochs

    def run_optimization(self, model, evaluation_param_bounds: dict,
                         num_top_results: int = 1):
        """
        Uses the BayesianOptimization library to find the best set of parameters given a range that generates the
        highest value of the model's metric.

        :param model: Needs to have a train method, a param reset method. The train method should output some metric
                      that the maximization_function wants to maximize.
        :param evaluation_param_bounds: The set of params that can be changed to maximize the model's performance.
                                        Beware, that some parameters could cause the algorithm to cheat such as
                                        increasing the number of epochs. These should only determine the architecture.
        :param data_bunch: A data bunch object for the model to train on.
        :param num_top_results: The number of results for this method to save. Defaults to a single top result
        :return: None, once the method is finished, the result's field should be populated.
        """

        def maximization_function(**params: dict):
            """
            The function whose value we want to maximize.

            :param params: The parameters to set to the model
            :return: The value of the metric used by the model to define its performance.
                     Expected to be the validation acc.
            """
            model.reset_params(params)

            return model.train(epochs=self.epochs, k=10)

        optimizer = BayesianOptimization(
            f=maximization_function,
            pbounds=evaluation_param_bounds,
            random_state=1,
        )

        logger = JSONLogger(path="./logs.json")
        optimizer.subscribe(Events.OPTMIZATION_END, logger)

        optimizer.maximize(
            init_points=0,
            n_iter=1,
        )

        sorted_results = sorted(optimizer.res, key=lambda k: k['target'])

        for parameter in list(reversed(sorted_results))[:num_top_results]:
            print(f'Keeping results: {parameter}')
            self.results.append(parameter)

    def batch_run(self):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        bayesian_optimizer = BayesianSearcher()
        model = CustomTabularModel(0.5, False, 1000, {'layer1': 20, 'layer2': 20})
        bayesian_optimizer.run_optimization(model, {'layer1': (1, 400), 'layer2': (1, 400), 'layer3': (1, 400)})
        bayesian_optimizer.run_optimization(model, {'layer1': (1, 400), 'layer2': (1, 400)})
        bayesian_optimizer.run_optimization(model, {'layer1': (1, 400)})

        print(str(bayesian_optimizer.results))

        now = datetime.now()
        log_path = os.path.join(str(Path(__file__).parents[0]), "logs")
        json.to_json(log_path + "/hyper_params" + now.strftime("%Y%m%d-%H%M%S.%f"), pd.DataFrame(bayesian_optimizer.results))

    @staticmethod
    def run_bayesian_search(epochs):
        if not os.path.exists(os.path.join(str(Path(__file__).parents[0]), "logs")):
            os.mkdir(os.path.join(str(Path(__file__).parents[0]), "logs"))

        bayesian_optimizer = BayesianSearcher(epochs)
        model = CustomTabularModel(0.5, False, 1000, {'layer1': 20, 'layer2': 20, 'dropout': 0.5})
        # bayesian_optimizer.run_optimization(model, {'layer1': (1, 400), 'layer2': (1, 400), 'layer3': (1, 400), 'dropout': 0.5})
        # bayesian_optimizer.run_optimization(model, {'layer1': (1, 400), 'layer2': (1, 400), 'dropout': 0.5})
        bayesian_optimizer.run_optimization(model, {'layer1': (1, 400), 'dropout': (0, 1)})

        print(str(bayesian_optimizer.results))

        now = datetime.now()
        log_path = os.path.join(str(Path(__file__).parents[0]), "logs")
        json.to_json(log_path + "/hyper_params" + now.strftime("%Y%m%d-%H%M%S.%f"), pd.DataFrame(bayesian_optimizer.results))
