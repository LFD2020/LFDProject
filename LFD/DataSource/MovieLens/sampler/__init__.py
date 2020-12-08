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

    def sample(self, batch_size):
        sampled_user_ids: _typing.List[int] = random.sample(
            self.__preprocessed_ratings.userId.drop_duplicates().to_list(), batch_size
        )

        def __sample_one_positive_movie(user_id: int) -> pd.DataFrame:
            __temp: pd.DataFrame = self.__preprocessed_ratings.loc[
                self.__preprocessed_ratings.userId == user_id
                ]
            __selected_row_index: int = random.randrange(0, len(__temp))
            return __temp.iloc[[__selected_row_index]]

        def __sample_one_negative_movie(user_id: int) -> pd.DataFrame:
            while True:
                __selected_movie_id: int = random.randrange(0, self.__all_movies_number)
                if len(
                        self.__preprocessed_ratings.loc[
                            (self.__preprocessed_ratings.userId == user_id) &
                            (self.__preprocessed_ratings.movieId == __selected_movie_id)
                        ]
                ) == 0:
                    return pd.DataFrame(
                        [[user_id, __selected_movie_id, 0.0, 0]],
                        columns=['userId', 'movieId', 'rating', 'timestamp']
                    )

        positive_samples: pd.DataFrame = pd.concat(
            [__sample_one_positive_movie(selected_user_id)
             for selected_user_id in sampled_user_ids],
            ignore_index=True
        )
        negative_samples: pd.DataFrame = pd.concat(
            [__sample_one_negative_movie(selected_user_id)
             for selected_user_id in sampled_user_ids],
            ignore_index=True
        )
        return positive_samples, negative_samples
