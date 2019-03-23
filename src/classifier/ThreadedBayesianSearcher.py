import itertools
import os
from datetime import datetime
import threading
from functools import partial
from pathlib import Path

from pandas.io.json import json
import pandas as pd

from src.classifier.BayesianTabularModelSearch import BayesianSearcher
from src.classifier.CustomTabularModel import CustomTabularModel


class ThreadedBayesianSearcher:

    @staticmethod
    def get_searchers(epochs, n_searchers, rand_points=0, iterations=2):
        optimizers = []

        for i in range(n_searchers):
            bayesian_optimizer = BayesianSearcher(epochs, rand_points=rand_points, iterations=iterations)
            optimizers.append(bayesian_optimizer)
        return optimizers

    @staticmethod
    def run(epochs, n_searchers = 2):
        p_bounds = [
            {'layer1': (1, 400), 'dropout': (0, 1)},
            {'layer1': (1, 400), 'layer2': (1, 400), 'layer3': (1, 400), 'dropout': (0, 1)},
            {'layer1': (1, 400), 'layer2': (1, 400), 'dropout': (0, 1)}
        ]
        p_bounds_iter = itertools.cycle(p_bounds)

        optimizers = ThreadedBayesianSearcher.get_searchers(epochs, n_searchers, 1, 1)

        targets = tuple([opt.run_optimization for opt in optimizers])

        optimizer_threads = []
        for target in targets:
            optimizer_threads.append(threading.Thread(target=partial(target,
                                                      evaluation_param_bounds=p_bounds_iter.__next__(),
                                                      num_top_results=2, k_folds=10)))
            # optimizer_threads[-1].daemon = True
            optimizer_threads[-1].start()

        for optimizer_thread in optimizer_threads:
            optimizer_thread.join()

        for result in BayesianSearcher.get_top_results_of_opts(optimizers, 2):
            print("found a maximum value of: {}".format(result['target']))

        now = datetime.now()
        log_path = os.path.join(str(Path(__file__).parents[0]), "logs")
        json.to_json(log_path + "/hyper_params" + now.strftime("%Y%m%d-%H%M%S.%f"),
                     pd.DataFrame(BayesianSearcher.get_top_results_of_opts(optimizers, 2)))