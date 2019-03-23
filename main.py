#!/usr/bin/python
"""
For running a single model do:
`main.py --train_model_test 10`


"""

import sys
import argparse
from src.classifier.CustomTabularModel import CustomTabularModel

if __name__ == '__main__':
    # Setup arguments
    parser = argparse.ArgumentParser(description='Upper level system for suicide prediction, and recommendation.')

    # Optional positional argument
    parser.add_argument('--train_model_test', type=int,
                        help='Run a model and specify the number of epochs')

    # Optional positional argument
    parser.add_argument('--run_bayes_model_generation', type=int,
                        help='Run the bayes model evaluation and specify the number of epochs')

    # Optional positional argument
    parser.add_argument('--run_recommendations', type=int,
                        help='Run the bayes model evaluation and specify the number of epochs')

    args = parser.parse_args()

    # Route Executions
    if args.train_model_test is not None:
        CustomTabularModel.test_model(args.train_model_test)