"""
This is an interface for all of the models to take advantage of.

Note: https://docs.fast.ai/tabular.html#tabular for information.


"""
from fastai.tabular import *
from pathlib import Path


class DataCsvInterface:
    """
        The absolute path to the csv. Recommend automatically setting this.

        As a note, the df that this will be using needs to be ALL NUMERIC

        ABS_CSV_PATH:     Basically, the abs path to the data directory
        CSV_FILE_NAME:    The name of the csv file. Can be a preprocessed version instead of original.
        DEPENDENT_NAME:   The y column to predict
        CATEGORY_NAMES:   The columns to handle categorically
        CONTINUOUS_NAMES: the columns to handled as continuous. If None, the non-cat, non-dep cols become continuous
    """
    ABS_CSV_PATH = str(Path(__file__).parents[0])
    CSV_FILE_NAME = "foreveralone_shuf.csv"
    DEPENDENT_NAME = "attempt_suicide"
    CATEGORY_NAMES = [e+"_categorical" for e in ['gender', 'sexuallity', 'race', 'virgin', 'prostitution_legal', 'pay_for_sex', 'social_fear', 'depressed', 'employment', 'edu_level', 'bodyweight', 'income']]
    CONTINUOUS_NAMES = ['age', 'income_float', 'friends']
    NF1_NAMES = ['what_help_from_others', 'improve_yourself_how']

    TRASH_NAMES = ['time', 'job_title']

    NF1_CONTINUOUS_NAMES = []
    NF1_CATEGORICAL_NAMES = ['improve_yourself_how', 'what_help_from_others']

    CATEGORY_NAMES += ['improve_yourself_how_'+str(i) for i in range(1,53+1)]
    CATEGORY_NAMES += ['what_help_from_others_' + str(i) for i in range(1, 45 + 1)]



    # needs to become continuous edu_level
    # need to preproc income avgerage on both sides of " to ", bodyweight
    # what_help_from_others, improve_yourself_how is 0nf
    # employment should be split
    VARIABLE_NAMES = ['income', 'bodyweight', 'prostitution_legal', 'pay_for_sex', 'friends', 'social_fear',
                      'depressed', 'employment', 'job_title', 'edu_level', 'what_help_from_others',
                      'improve_yourself_how', 'virgin']
    FIXED_NAMES = ['gender', 'sexuallity', 'age', 'race']

    @staticmethod
    def get_data_info(split_percent=0.5, randomize=False, n_rows=None) -> DataBunch:
        """
        The method that the entire classifier system will call to train, validate, and test on.

        :param n_rows:
        :param split_percent:
        :param randomize:
        :return:
        """
        train_val_df = pd.read_csv(os.path.join(DataCsvInterface.ABS_CSV_PATH, DataCsvInterface.CSV_FILE_NAME),
                                   nrows=n_rows)
        # When the train_val_df is loaded, you can split the train_val_df and add it to the
        # test train_val_df or load a separate one
        test_df = None

        """ Get the indices and split them. The missing indices become the validation set. Optionally shuffle. """
        valid_idx = list(range(0, len(train_val_df)))
        if randomize:
            random.shuffle(valid_idx)
        valid_idx = valid_idx[int(len(valid_idx) * split_percent):]

        """ Set the processes to run on the dfs """
        transforms = [FillMissing, Categorify, Normalize]

        batch_size = 32 if len(valid_idx) < 800 else 64

        # noinspection PyTypeChecker
        train_test_validate_data = TabularDataBunch.from_df(DataCsvInterface.ABS_CSV_PATH, train_val_df,
                                                            DataCsvInterface.DEPENDENT_NAME, valid_idx=valid_idx,
                                                            procs=transforms,
                                                            test_df=test_df,
                                                            cat_names=DataCsvInterface.CATEGORY_NAMES,
                                                            cont_names=DataCsvInterface.CONTINUOUS_NAMES,
                                                            bs=batch_size)
        return train_test_validate_data
