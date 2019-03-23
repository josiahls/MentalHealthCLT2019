import os
from datetime import datetime
from pathlib import Path

from pandas.io.json import json
import pandas as pd
from bayes_opt import BayesianOptimization, Events, JSONLogger
from fastai.basic_data import DataBunch, Tensor
import numpy as np

from src.classifier.CustomTabularModel import CustomTabularModel
from src.data.DataCsvInterface import DataCsvInterface
from src.recommender.JSONParamReader import JSONParamReader


class BayesianRecommender:
    def __init__(self):
        self.results = []

    def run_optimization(self, model, data, evaluation_param_bounds: dict,
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
            # Fill in any missing data for running predictions
            for key in data:
                if key not in params:
                    params[key] = data[key]
            # If any variables are being shifted back, then PUNISH
            for key in [_ for _ in params if _ in DataCsvInterface.ONE_WAY_NAMES]:
                if params[key] > data[key]:
                    print('Punishing')
                    return 0

            # Note we are trying to maximize the first label (not commit suicide)
            maximizing_value = float(model.predict(params)[2][0])
            print(f'Value to maximize: {maximizing_value}')

            return maximizing_value

        optimizer = BayesianOptimization(
            f=maximization_function,
            pbounds=evaluation_param_bounds,
            verbose=2,
            random_state=1,
        )

        # Starting value:
        maximizing_value = float(model.predict(data)[2][0])
        print(f'Value to maximize: {maximizing_value}')

        # optimizer.probe(params=data)

        log_path = os.path.join(str(Path(__file__).parents[0]), "logs")
        logger = JSONLogger(path=log_path + "/raw_logs.json")
        optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=2,
            n_iter=3,
        )

        sorted_results = sorted(optimizer.res, key=lambda k: k['target'])

        for parameter in list(reversed(sorted_results))[:num_top_results]:
            print(f'Keeping results: {parameter}')
            self.results.append(parameter)

    def get_ranges(self, names, datalist):
        cat_max = np.max(datalist.x.codes, axis=1)
        cat_min = np.min(datalist.x.codes, axis=1)
        con_max = np.max(datalist.x.conts, axis=1)
        con_min = np.min(datalist.x.conts, axis=1)

        new_cat_ranges = {key: (cat_min[i], cat_max[i]) for i, key in enumerate(names[:len(cat_max)])}
        new_con_ranges = {key: (con_min[i], con_max[i]) for i, key in enumerate(names[len(cat_max):])}

        return dict(new_cat_ranges, **new_con_ranges)

    @staticmethod
    def test_recommender():
        if not os.path.exists(os.path.join(str(Path(__file__).parents[0]), "logs")):
            os.mkdir(os.path.join(str(Path(__file__).parents[0]), "logs"))
        if not os.path.exists(os.path.join(str(Path(__file__).parents[0]), "raw_logs")):
            os.mkdir(os.path.join(str(Path(__file__).parents[0]), "raw_logs"))

        bayesian_optimizer = BayesianRecommender()
        # Init the model with the best params
        model = CustomTabularModel(0.5, False, 1000, {'layer1': 20, 'layer2': 20})
        best_params = JSONParamReader('classifier/logs').get_best_param()
        model.reset_params(best_params)

        data = model.input_data.train_ds[0][0]
        data_parsed = []
        for element in data.data:
            if type(element) is Tensor:
                data_parsed += list(element.numpy())
            else:
                data_parsed += element

        data_init = {key: data_parsed[i] for i, key in enumerate(data.names)}
        cr = bayesian_optimizer.get_ranges(data.names, model.input_data.train_ds)
        column_range = {key: cr[key] for key in cr if key not in DataCsvInterface.FIXED_NAMES}

        model.train(10)

        bayesian_optimizer.run_optimization(model, data_init, column_range)

        print(str(bayesian_optimizer.results))

        now = datetime.now()
        log_path = os.path.join(str(Path(__file__).parents[0]), "logs")
        json.to_json(log_path + "/hyper_params" + now.strftime("%Y%m%d-%H%M%S.%f") + ".json", pd.DataFrame(bayesian_optimizer.results))
        model.input_data.save(log_path + "/input_information" + now.strftime("%Y%m%d-%H%M%S.%f"))