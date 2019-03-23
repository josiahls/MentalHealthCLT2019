import os
from pandas.io.json import json
from pathlib import Path


class JSONParamReader:

    def __init__(self, log_path: str = 'logs'):
        self.log_path = log_path

    def get_params(self):
        files = []
        abs_path = os.path.join(str(Path(__file__).parents[1]), self.log_path)
        for file in os.listdir(abs_path):
            files.append(file)

        return json.read_json(os.path.join(abs_path, sorted(files)[-1]))

    def get_best_param(self):
        df = self.get_params()
        return df.loc[df['target'].argmax()]['params']


if __name__ == '__main__':
    json_data = JSONParamReader('classifier/logs').get_best_param()
    print(json_data)
