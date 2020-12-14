import os
import pickle
import multiprocessing
import typing as _typing

import numpy as np
import pandas as pd


class MovieLensDataPreprocessor:
    def __init__(self, path_to_data_directory: str):
        """
        :param path_to_data_directory: path to directory containing MovieLens data
        """
        self.__data_directory_absolute_path: str = os.path.abspath(
            os.path.expanduser(path_to_data_directory)
        )
        for __filename in ('movies.dat', 'users.dat', 'ratings.dat'):
            if not os.path.isfile(os.path.join(self.__data_directory_absolute_path, __filename)):
                raise FileNotFoundError(
                    "\'%s\' NOT found or not a file" % os.path.join(
                        self.__data_directory_absolute_path, __filename
                    )
                )
        self.__raw_movies_data: pd.DataFrame = pd.read_csv(
            os.path.join(self.__data_directory_absolute_path, 'movies.dat'),
            sep='::', header=None, names=['movieId', 'movieTitle', 'movieGenre'],
            engine='python'
        )
        self.__raw_users_data: pd.DataFrame = pd.read_csv(
            os.path.join(self.__data_directory_absolute_path, 'users.dat'),
            sep='::', header=None, names=['userId', 'gender', 'age', 'occupation', 'zipCode'],
            engine='python'
        )
        self.__raw_ratings_data: pd.DataFrame = pd.read_csv(
            os.path.join(self.__data_directory_absolute_path, 'ratings.dat'),
            sep='::', header=None, names=['userId', 'movieId', 'rating', 'timestamp'],
            engine='python'
        )

        self.__MovieGenresMap: _typing.Dict[str, int] = {
            'Action': 0,
            'Adventure': 1,
            'Animation': 2,
            'Children\'s': 3,
            'Comedy': 4,
            'Crime': 5,
            'Documentary': 6,
            'Drama': 7,
            'Fantasy': 8,
            'Film-Noir': 9,
            'Horror': 10,
            'Musical': 11,
            'Mystery': 12,
            'Romance': 13,
            'Sci-Fi': 14,
            'Thriller': 15,
            'War': 16,
            'Western': 17
        }
        self.__AgesMap: _typing.Dict[int, int] = {
            1: 0,
            18: 1,
            25: 2,
            35: 3,
            45: 4,
            50: 5,
            56: 6
        }

    def process(self):
        def __process_movies(
                raw_movies_data: pd.DataFrame,
                genres_map: _typing.Dict[str, int]
        ) -> pd.DataFrame:
            __preprocessed_movies_data: pd.DataFrame = raw_movies_data
            __preprocessed_movies_data['mappedId'] = range(len(raw_movies_data))
            __preprocessed_movies_data['genres'] = [
                np.zeros(1) for _ in range(len(raw_movies_data))
            ]
            '''
            To be more generous,
            the map of genres should be adaptively generated according to given data
            '''
            for __i in range(len(__preprocessed_movies_data)):
                __movie_genres_str: str = __preprocessed_movies_data.iloc[__i, 2]
                __preprocessed_movies_data.iat[__i, 4] = np.array([
                    genres_map.get(__genre_str)
                    for __genre_str in __movie_genres_str.split('|')
                ])
            return __preprocessed_movies_data

        def __process_users(
                raw_users_data: pd.DataFrame,
                ages_map: _typing.Dict[int, int]
        ) -> pd.DataFrame:
            __preprocessed_users_data: pd.DataFrame = pd.DataFrame(
                {
                    'userId': range(len(raw_users_data)),
                    'occupation': raw_users_data.iloc[:, 3]
                },
                columns=['userId', 'gender', 'age', 'occupation']
            )
            for __i in range(len(raw_users_data)):
                __preprocessed_users_data.iloc[__i, 1] = \
                    0 if raw_users_data.iloc[__i, 1] == 'F' else 1
                __preprocessed_users_data.iloc[__i, 2] = \
                    ages_map.get(raw_users_data.iloc[__i, 2])
            return __preprocessed_users_data

        __preprocessed_movies: pd.DataFrame = __process_movies(
            self.__raw_movies_data, self.__MovieGenresMap
        )
        __preprocessed_users: pd.DataFrame = __process_users(
            self.__raw_users_data, self.__AgesMap
        )

        ''' remap ratings '''
        __preprocessed_ratings: pd.DataFrame = pd.DataFrame(
            {
                'userId': self.__raw_ratings_data.userId - 1,
                'rating': self.__raw_ratings_data.rating,
                'timestamp': self.__raw_ratings_data.timestamp
            },
            columns=['userId', 'movieId', 'rating', 'timestamp']
        )
        import tqdm
        with tqdm.tqdm(range(len(self.__raw_ratings_data))) as t:
            for i in t:
                __original_movie_id = self.__raw_ratings_data.iat[i, 1]
                __temp_data_frame: pd.DataFrame = __preprocessed_movies.loc[
                    __preprocessed_movies['movieId'] == __original_movie_id
                ]
                assert len(__temp_data_frame) == 1
                __preprocessed_ratings.iat[i, 1] = __temp_data_frame.iat[0, 3]
        return __preprocessed_users, __preprocessed_movies, __preprocessed_ratings

    def save(self, users_movies_ratings: _typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):
        with open(os.path.join(self.__data_directory_absolute_path, "preprocessed_data.pkl"), "wb") as file:
            pickle.dump(users_movies_ratings, file)
