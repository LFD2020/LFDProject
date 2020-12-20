import random
import typing as _typing
import pandas as pd


class MovieLensSampler:
    def __init__(
            self, preprocessed_ratings: pd.DataFrame,
            all_users_number: int, all_movies_number: int
    ):
        if not isinstance(preprocessed_ratings, pd.DataFrame):
            raise TypeError

        self.__preprocessed_ratings: pd.DataFrame = preprocessed_ratings
        self.__all_users_number: int = all_users_number
        self.__all_movies_number: int = all_movies_number

    @property
    def number_of_ratings(self) -> int:
        return len(self.__preprocessed_ratings)

    def __sample_with_single_thread(self, batch_size: int) -> _typing.Tuple[pd.DataFrame, pd.DataFrame]:

        def __every_user_sample(
                number_of_items_per_user: int
        ) -> _typing.Tuple[pd.DataFrame, pd.DataFrame]:
            if not number_of_items_per_user > 0:
                raise ValueError
            __positive_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
            __negative_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
            for user_id in self.__preprocessed_ratings['userId'].drop_duplicates():
                """ Generate positive samples """
                __movies_of_current_user: pd.DataFrame = self.__preprocessed_ratings.loc[
                    self.__preprocessed_ratings.userId == user_id
                    ]
                __positive_ratings: pd.DataFrame = __positive_ratings.append(
                    __movies_of_current_user.sample(number_of_items_per_user),
                    ignore_index=True
                )

                """ Generate negative samples """
                __negative_items_of_current_user: pd.DataFrame = pd.DataFrame(
                    columns=['userId', 'movieId', 'rating', 'timestamp']
                )
                while len(__negative_items_of_current_user) < number_of_items_per_user:
                    __selected_movie_id: int = random.randrange(0, self.__all_movies_number)
                    if len(__movies_of_current_user.loc[__movies_of_current_user.movieId == __selected_movie_id]) == 0:
                        __negative_items_of_current_user: pd.DataFrame = __negative_items_of_current_user.append(
                            pd.DataFrame({
                                'userId': [user_id],
                                'movieId': [__selected_movie_id],
                                'rating': [0],
                                'timestamp': [0]
                            }),
                            ignore_index=True
                        )
                __negative_ratings: pd.DataFrame = __negative_ratings.append(
                    __negative_items_of_current_user, ignore_index=True
                )
            return __positive_ratings, __negative_ratings

        if batch_size // self.__all_users_number > 0:
            positive_ratings, negative_ratings = __every_user_sample(batch_size // self.__all_users_number)
        else:
            positive_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
            negative_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

        while len(positive_ratings) < batch_size or len(negative_ratings) < batch_size:
            assert len(positive_ratings) == len(negative_ratings)
            __selected_users: _typing.List[int] = random.sample(
                range(self.__all_users_number),
                batch_size - len(positive_ratings)
            )
            for __current_selected_user_id in __selected_users:
                movies_of_current_user: pd.DataFrame = self.__preprocessed_ratings.loc[
                    self.__preprocessed_ratings.userId == __current_selected_user_id
                    ]
                unselected_candidates: pd.DataFrame = movies_of_current_user.append(
                    positive_ratings.loc[positive_ratings.userId == __current_selected_user_id],
                    ignore_index=True
                ).drop_duplicates(['userId', 'movieId', 'rating'], keep=False)

                if len(unselected_candidates) == 0:
                    continue

                positive_ratings: pd.DataFrame = positive_ratings.append(
                    unselected_candidates.sample(1), ignore_index=True
                )
                while True:
                    __random_movie_id: int = random.randrange(0, self.__all_movies_number)
                    if len(movies_of_current_user.loc[movies_of_current_user.movieId == __random_movie_id]) > 0:
                        continue
                    else:
                        negative_ratings: pd.DataFrame = negative_ratings.append(
                            pd.DataFrame({
                                'userId': [__current_selected_user_id],
                                'movieId': [__random_movie_id],
                                'rating': [0],
                                'timestamp': [0]
                            }),
                            ignore_index=True
                        )
                        break
                if len(positive_ratings) >= batch_size or len(negative_ratings) >= batch_size:
                    break
        assert len(positive_ratings) == len(negative_ratings) == batch_size
        return positive_ratings, negative_ratings

    def sample(self, batch_size: int) -> _typing.Tuple[pd.DataFrame, pd.DataFrame]:
        import os
        import multiprocessing

        def __every_user_sample(
                number_of_items_per_user: int
        ) -> _typing.Tuple[pd.DataFrame, pd.DataFrame]:
            def __slice_by_range_of_users(start_user_id: int, stop_user_id: int) -> pd.DataFrame:
                return self.__preprocessed_ratings.loc[
                    (self.__preprocessed_ratings.userId >= start_user_id) &
                    (self.__preprocessed_ratings.userId < stop_user_id)
                ]

            def __sample_on_data_slice(__data_slice: pd.DataFrame, __number_of_items_per_user: int):
                if not number_of_items_per_user > 0:
                    raise ValueError
                __positive_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
                __negative_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
                for __user_id in __data_slice['userId'].drop_duplicates():
                    """ Generate positive samples """
                    __movies_of_current_user: pd.DataFrame = __data_slice.loc[
                        __data_slice.userId == __user_id
                    ]
                    __positive_ratings: pd.DataFrame =

            if not number_of_items_per_user > 0:
                raise ValueError
            __positive_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
            __negative_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
            for user_id in self.__preprocessed_ratings['userId'].drop_duplicates():
                """ Generate positive samples """
                __movies_of_current_user: pd.DataFrame = self.__preprocessed_ratings.loc[
                    self.__preprocessed_ratings.userId == user_id
                    ]
                __positive_ratings: pd.DataFrame = __positive_ratings.append(
                    __movies_of_current_user.sample(number_of_items_per_user),
                    ignore_index=True
                )

                """ Generate negative samples """
                __negative_items_of_current_user: pd.DataFrame = pd.DataFrame(
                    columns=['userId', 'movieId', 'rating', 'timestamp']
                )
                while len(__negative_items_of_current_user) < number_of_items_per_user:
                    __selected_movie_id: int = random.randrange(0, self.__all_movies_number)
                    if len(__movies_of_current_user.loc[__movies_of_current_user.movieId == __selected_movie_id]) == 0:
                        __negative_items_of_current_user: pd.DataFrame = __negative_items_of_current_user.append(
                            pd.DataFrame({
                                'userId': [user_id],
                                'movieId': [__selected_movie_id],
                                'rating': [0],
                                'timestamp': [0]
                            }),
                            ignore_index=True
                        )
                __negative_ratings: pd.DataFrame = __negative_ratings.append(
                    __negative_items_of_current_user, ignore_index=True
                )
            return __positive_ratings, __negative_ratings

        if batch_size // self.__all_users_number > 0:
            positive_ratings, negative_ratings = __every_user_sample(batch_size // self.__all_users_number)
        else:
            positive_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
            negative_ratings: pd.DataFrame = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

        while len(positive_ratings) < batch_size or len(negative_ratings) < batch_size:
            assert len(positive_ratings) == len(negative_ratings)
            __selected_users: _typing.List[int] = random.sample(
                range(self.__all_users_number),
                batch_size - len(positive_ratings)
            )
            for __current_selected_user_id in __selected_users:
                movies_of_current_user: pd.DataFrame = self.__preprocessed_ratings.loc[
                    self.__preprocessed_ratings.userId == __current_selected_user_id
                    ]
                unselected_candidates: pd.DataFrame = movies_of_current_user.append(
                    positive_ratings.loc[positive_ratings.userId == __current_selected_user_id],
                    ignore_index=True
                ).drop_duplicates(['userId', 'movieId', 'rating'], keep=False)

                if len(unselected_candidates) == 0:
                    continue

                positive_ratings: pd.DataFrame = positive_ratings.append(
                    unselected_candidates.sample(1), ignore_index=True
                )
                while True:
                    __random_movie_id: int = random.randrange(0, self.__all_movies_number)
                    if len(movies_of_current_user.loc[movies_of_current_user.movieId == __random_movie_id]) > 0:
                        continue
                    else:
                        negative_ratings: pd.DataFrame = negative_ratings.append(
                            pd.DataFrame({
                                'userId': [__current_selected_user_id],
                                'movieId': [__random_movie_id],
                                'rating': [0],
                                'timestamp': [0]
                            }),
                            ignore_index=True
                        )
                        break
                if len(positive_ratings) >= batch_size or len(negative_ratings) >= batch_size:
                    break
        assert len(positive_ratings) == len(negative_ratings) == batch_size
        return positive_ratings, negative_ratings
