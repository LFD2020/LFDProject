import math
import random
import pandas as pd
import torch
try:
    import tqdm as tqd
except ModuleNotFoundError:
    tqd = None

import typing as _typing

''' References for PyTorch sparse supports:
https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html
'''


class RemapIndices:
    def __call__(self, ratings: pd.DataFrame) -> pd.DataFrame:
        __user_id_map: dict = {}
        __movie_id_map: dict = {}
        '''
        __intermediate_data: pd.DataFrame = ratings.sort_values(['userId', 'movieId'])
        for row in __intermediate_data.itertuples(index=False):
            if row.userId not in __user_id_map:
                __new_id: int = len(__user_id_map)
                __user_id_map[row.userId] = __new_id
        __intermediate_data: pd.DataFrame = ratings.sort_values(['movieId', 'userId'])
        for row in __intermediate_data.itertuples(index=False):
            if row.movieId not in __movie_id_map:
                __new_id: int = len(__movie_id_map)
                __movie_id_map[row.movieId] = __new_id
        '''
        for user_id in ratings['userId'].drop_duplicates().sort_values():
            if user_id not in __user_id_map:
                __user_id_map[user_id] = len(__user_id_map)
        for movie_id in ratings['movieId'].drop_duplicates().sort_values():
            if movie_id not in __movie_id_map:
                __movie_id_map[movie_id] = len(__movie_id_map)

        for i in ratings.index:
            ratings.loc[i, 'userId'] = __user_id_map.get(ratings.loc[i, 'userId'])
            ratings.loc[i, 'movieId'] = __movie_id_map.get(ratings.loc[i, 'movieId'])

        return ratings


class MovieLensRatingsDataSplitter:
    @classmethod
    def _split_for_one_user(
            cls, __one_user_ratings: pd.DataFrame,
            __items_number_per_user_into_test_set
    ) -> _typing.Tuple[pd.DataFrame, pd.DataFrame]:
        __one_user_ratings: pd.DataFrame = \
            __one_user_ratings.sort_values('timestamp', axis=0, ascending=False)
        _current_user_test_set: pd.DataFrame = \
            __one_user_ratings.iloc[
                range(__items_number_per_user_into_test_set)
            ]
        _current_user_train_set: pd.DataFrame = \
            __one_user_ratings.iloc[
                range(__items_number_per_user_into_test_set, len(__one_user_ratings))
            ]
        return _current_user_train_set, _current_user_test_set

    def __call__(
            self, preprocessed_all_ratings: pd.DataFrame,
            items_number_per_user_into_test_set: int = 1
    ) -> _typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Utilize the Leave-n-out strategy
        to split all ratings into train_set and test_set
        with multiprocessing
        :param preprocessed_all_ratings: preprocessed all ratings
        :param items_number_per_user_into_test_set: items number per user into test set
        :return: train set and test set
        """
        if not (
                isinstance(preprocessed_all_ratings, pd.DataFrame) and
                type(items_number_per_user_into_test_set) == int
        ):
            raise TypeError
        if not items_number_per_user_into_test_set > 0:
            raise ValueError

        import multiprocessing
        with multiprocessing.Pool() as pool:
            __async_results: _typing.List[multiprocessing.pool.AsyncResult] = [
                pool.apply_async(
                    self._split_for_one_user,
                    (preprocessed_all_ratings.loc[preprocessed_all_ratings.userId == __user_id],
                     items_number_per_user_into_test_set)
                ) for __user_id in preprocessed_all_ratings['userId'].drop_duplicates()
            ]
            pool.close()
            pool.join()
            __train_set: pd.DataFrame = \
                pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
            __test_set: pd.DataFrame = \
                pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
            for __async_result in __async_results:
                __train_set = __train_set.append(__async_result.get()[0], ignore_index=True)
                __test_set = __test_set.append(__async_result.get()[1], ignore_index=True)
            __train_set: pd.DataFrame = __train_set.sort_values(['userId', 'movieId'])
            __test_set: pd.DataFrame = __test_set.sort_values(['userId', 'movieId'])
            return __train_set, __test_set


# class ToTrainSetAndTestSet:
#     def __call__(self, ratings: pd.DataFrame) -> _typing.Tuple[pd.DataFrame, pd.DataFrame]:
#         train_set: pd.DataFrame = pd.DataFrame(
#             columns=['userId', 'movieId', 'rating', 'timestamp']
#         )
#         test_set: pd.DataFrame = pd.DataFrame(
#             columns=['userId', 'movieId', 'rating', 'timestamp']
#         )
#         user_list: _typing.List[int] = ratings['userId'].drop_duplicates().to_list()
#         for user_id in tqd.tqdm(user_list) if tqd is not None else user_list:
#             current_user_ratings: pd.DataFrame = ratings[ratings.userId == user_id]
#             number_of_movies_in_test_set: int = len(current_user_ratings) // 5
#             if len(current_user_ratings) % 5 > 0:
#                 if (number_of_movies_in_test_set + 1) / len(current_user_ratings) < 0.25:
#                     number_of_movies_in_test_set += 1
#             indices_for_train_set: list = random.sample(
#                 range(len(current_user_ratings)),
#                 len(current_user_ratings) - number_of_movies_in_test_set
#             )
#             for index in range(len(current_user_ratings)):
#                 if index in indices_for_train_set:
#                     train_set = train_set.append(
#                         current_user_ratings.iloc[[index]], ignore_index=True
#                     )
#                 else:
#                     test_set = test_set.append(
#                         current_user_ratings.iloc[[index]], ignore_index=True
#                     )
#         return train_set, test_set


class MatricesGenerator:
    def __init__(
            self, ratings: pd.DataFrame,
            all_users_number: int, all_movies_number: int
    ):
        if not isinstance(ratings, pd.DataFrame):
            raise TypeError

        __indexes: list = []
        __ratings: list = []
        for row in ratings.itertuples(index=False):
            __indexes.append([row.userId, row.movieId])
            __ratings.append(row.rating)

        self.__interaction_matrix: torch.sparse.Tensor = torch.sparse_coo_tensor(
            torch.tensor(__indexes).t(), __ratings,
            size=(all_users_number, all_movies_number)
        )

        __user_neighbors_number: dict = {}
        for user_id in ratings['userId'].drop_duplicates().to_list():
            __user_neighbors_number[user_id] = len(ratings[ratings.userId == user_id])

        __item_neighbors_number: dict = {}
        for movie_id in ratings['movieId'].drop_duplicates().to_list():
            __item_neighbors_number[movie_id] = len(ratings[ratings.movieId == movie_id])

        __intermediate_values = [
            1.0 / (
                    math.sqrt(__user_neighbors_number.get(user_id)) *
                    math.sqrt(__item_neighbors_number.get(movie_id))
            )
            for user_id, movie_id in __indexes
        ]

        __top_right_indices: torch.Tensor = torch.tensor(__indexes, dtype=torch.int32).t()
        __top_right_indices[1] += all_users_number
        __bottom_left_indices: torch.Tensor = torch.tensor(__indexes, dtype=torch.int32).t()[[1, 0]]
        __bottom_left_indices[0] += all_users_number

        self.__normalized_l: torch.sparse.Tensor = torch.sparse_coo_tensor(
            torch.cat((__top_right_indices, __bottom_left_indices), dim=1),
            torch.tensor(__intermediate_values).repeat(2),
            size=(
                all_users_number + all_movies_number,
                all_users_number + all_movies_number
            )
        )

    @property
    def normalized_l(self) -> torch.sparse.Tensor:
        return self.__normalized_l

    @property
    def interaction_matrix(self) -> torch.sparse.Tensor:
        return self.__interaction_matrix
