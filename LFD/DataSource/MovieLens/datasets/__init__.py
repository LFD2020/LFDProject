import os
import pandas as pd
import typing as _typing


class _MovieLensData:

    def __init__(self, directory_path: str):
        if type(directory_path) != str:
            raise TypeError
        directory_path: str = os.path.expanduser(directory_path)
        if os.path.isabs(directory_path):
            self._data_directory_absolute_path: str = directory_path
        else:
            self._data_directory_absolute_path: str = os.path.join(
                self.package_absolute_path(), directory_path
            )
        if not os.path.isdir(self._data_directory_absolute_path):
            raise NotADirectoryError
        for file in [
            os.path.join(self._data_directory_absolute_path, filename)
            for filename in ("movies.csv", "ratings.csv")
        ]:
            if not os.path.isfile(file):
                raise FileNotFoundError("%s not found or not a file" % file)

        self.__ratings: _typing.Optional[pd.DataFrame] = pd.read_csv(
            os.path.join(self._data_directory_absolute_path, "ratings.csv")
        )

    @property
    def number_of_users(self) -> int:
        return len(self.__ratings['userId'].drop_duplicates())

    @property
    def number_of_movies(self) -> int:
        return len(self.__ratings['movieId'].drop_duplicates())

    @property
    def original_ratings(self) -> pd.DataFrame:
        return self.__ratings

    def load_train_set(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self._data_directory_absolute_path, "train.csv"))

    def save_train_set(self, train_set: pd.DataFrame):
        train_set.to_csv(
            os.path.join(self._data_directory_absolute_path, "train.csv"), index=False
        )

    def load_test_set(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self._data_directory_absolute_path, "test.csv"))

    def save_test_set(self, test_set: pd.DataFrame):
        test_set.to_csv(
            os.path.join(self._data_directory_absolute_path, "test.csv"), index=False
        )

    @classmethod
    def package_absolute_path(cls) -> str:
        """
        Get the absolute path to this package(directory)
        :return (str): the absolute path to this package(directory)
        """
        from . import _utils
        return os.path.split(_utils.__file__)[0]
