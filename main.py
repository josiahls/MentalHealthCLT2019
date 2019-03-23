#!/usr/bin/python
"""
For running a single model do:
`main.py --train_model_test 10`

For running the bayesian model selection do:
`main.py --run_bayes_model_generation 10`

For running the bayesian recommender do:
`main.py --run_recommendations`


"""

import sys
import argparse

from src.classifier.ThreadedBayesianSearcher import ThreadedBayesianSearcher
from src.classifier.BayesianTabularModelSearch import BayesianSearcher
from src.classifier.CustomTabularModel import CustomTabularModel
from src.recommender.BayesianRecommender import BayesianRecommender

if __name__ == '__main__':
    # Setup arguments
    parser = argparse.ArgumentParser(description='Upper level system for suicide prediction, and recommendation.')

    group = parser.add_mutually_exclusive_group(required=True)
    # Optional positional argument
    group.add_argument('--train_model_test', type=int,
                       help='Run a model and specify the number of epochs')

    # Optional positional argument
    group.add_argument('--run_bayes_model_generation', type=int,
                       help='Run the bayes model evaluation and specify the number of epochs')

    # Optional positional argument
    group.add_argument('--run_recommendations', type=str,
                       help='Run the recommendation engine, add "run"')
    # Optional positional argument
    group.add_argument('--run_threaded_bayes_model_generation', type=int,
                       help='Run the recommendation engine, add "run"')

    args = parser.parse_args()

    # Route Executions
    if args.train_model_test is not None:
        CustomTabularModel.test_model(args.train_model_test)
    elif args.run_bayes_model_generation is not None:
        BayesianSearcher.run_bayesian_search(args.run_bayes_model_generation)
    elif args.run_recommendations is not None:
        BayesianRecommender.test_recommender()
    elif args.run_threaded_bayes_model_generation is not None:
        ThreadedBayesianSearcher.run(args.run_threaded_bayes_model_generation)

