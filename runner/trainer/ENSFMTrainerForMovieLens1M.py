import os
import pickle
import typing as _typing

import pandas as pd
import torch.utils.data
import tqdm

from model.ENSFM import ENSFMWithEmbeddings
from weights import TorchWeightsUtils


class _TrainingDataSet(torch.utils.data.Dataset):
    @classmethod
    def _convert_ratings_for_one_user(
            cls, ratings_of_one_user: pd.DataFrame
    ) -> _typing.Tuple[int, _typing.List[int]]:
        if len(ratings_of_one_user['userId'].drop_duplicates()) != 1:
            raise ValueError
        return (
            ratings_of_one_user['userId'].drop_duplicates().tolist()[0],
            ratings_of_one_user['movieId'].drop_duplicates().tolist()
        )

    def __init__(self, ratings_data: pd.DataFrame):
        def _convert_ratings(
                preprocessed_ratings: pd.DataFrame
        ) -> _typing.List[_typing.Tuple[int, _typing.List[int]]]:
            import multiprocessing
            with multiprocessing.Pool() as pool:
                __async_results: _typing.Dict[
                    int, multiprocessing.pool.AsyncResult
                ] = {}
                for __user_id in preprocessed_ratings['userId'].drop_duplicates():
                    __async_results[__user_id] = pool.apply_async(
                        self._convert_ratings_for_one_user,
                        (preprocessed_ratings.loc[preprocessed_ratings.userId == __user_id],)
                    )
                pool.close()
                pool.join()
                return [__async_results[k].get() for k in sorted(__async_results.keys())]

        self.__all_users_with_positive_items: _typing.List[
            _typing.Tuple[int, _typing.List[int]]
        ] = _convert_ratings(ratings_data)

    def __len__(self) -> int:
        return len(self.__all_users_with_positive_items)

    def __getitem__(self, idx: int) -> _typing.Tuple[int, _typing.List[int]]:
        return self.__all_users_with_positive_items[idx]


class ENSFMTrainerForMovieLens1M:
    """ Trainer for Efficient Non-Sampling Factorization Machines on MovieLens-1M """

    def __init__(self, data_directory_path: str):
        data_directory_path: str = os.path.abspath(
            os.path.expanduser(data_directory_path)
        )
        for required_preprocessed_data_filename in (
                'preprocessed_data.pkl', 'train.csv', 'test.csv'
        ):
            if not os.path.isfile(
                    os.path.join(data_directory_path, required_preprocessed_data_filename)
            ):
                raise FileNotFoundError
        self.__data_directory_path: str = data_directory_path

        self.__preprocessed_users_data: pd.DataFrame
        self.__preprocessed_movies_data: pd.DataFrame
        self.__ratings_train_set: pd.DataFrame

        ''' load intermediate data '''
        with open(os.path.join(self.__data_directory_path, 'preprocessed_data.pkl'), 'rb') as file:
            __intermediate_data = pickle.load(file)
            self.__preprocessed_users_data: pd.DataFrame = __intermediate_data[0]
            self.__preprocessed_movies_data: pd.DataFrame = __intermediate_data[1]
        __number_of_users: int = len(self.__preprocessed_users_data)
        __number_of_age_groups: int = 7
        __number_of_occupations: int = 21
        __number_of_movies: int = len(self.__preprocessed_movies_data)
        __number_of_genres: int = 18
        self.__model: ENSFMWithEmbeddings = ENSFMWithEmbeddings(
            __number_of_users + 2 + __number_of_age_groups + __number_of_occupations,
            __number_of_movies + __number_of_genres, 64
        )
        self.__all_users_nonzero_indexes_of_features: _typing.List[
            _typing.Tuple[int, _typing.List[int]]
        ] = []
        for __user_index in range(__number_of_users):
            __user_id = self.__preprocessed_users_data.iat[__user_index, 0]
            __gender = self.__preprocessed_users_data.iat[__user_index, 1]
            __gender += __number_of_users
            __age = self.__preprocessed_users_data.iat[__user_index, 2]
            __age += __number_of_users + 2
            __occupation = self.__preprocessed_users_data.iat[__user_index, 3]
            __occupation += __number_of_users + 2 + 7
            self.__all_users_nonzero_indexes_of_features.append(
                (__user_id, [__user_id, __gender, __age, __occupation])
            )
        self.__all_items_nonzero_indexes_of_features: _typing.List[
            _typing.Tuple[int, _typing.List[int]]
        ] = []
        for _movie_index in range(__number_of_movies):
            _movie_id = self.__preprocessed_movies_data.iat[_movie_index, 3]
            _movie_genres = self.__preprocessed_movies_data.iat[_movie_index, 4]
            _movie_genres += __number_of_movies
            self.__all_items_nonzero_indexes_of_features.append(
                (_movie_id, [_movie_id] + _movie_genres.tolist())
            )
        ''' load train set '''
        self.__ratings_train_set: pd.DataFrame = pd.read_csv(
            os.path.join(self.__data_directory_path, 'train.csv')
        )

    @classmethod
    def __compute_loss(
            cls, h2: torch.Tensor,
            negative_weight: float,
            all_items_q_matrix: torch.Tensor,
            all_users_p_matrix: torch.Tensor,
            batch_users_with_positive_items: _typing.List[
                _typing.Tuple[int, _typing.List[int]]
            ]
    ) -> torch.Tensor:
        batch_users_p_matrix: torch.Tensor = all_users_p_matrix[
            [
                one_user_with_positive_items[0]
                for one_user_with_positive_items in batch_users_with_positive_items
            ]
        ]
        loss_function = ENSFMWithEmbeddings.Loss(torch.nn.Dropout(0.1))
        return loss_function(
            h2, negative_weight,
            all_items_q_matrix,
            batch_users_p_matrix,
            batch_users_with_positive_items
        )

    def train(self):
        __epochs: int = 1000
        __users_batch_size: int = 512
        __weights_directory_name: str = "E" + "NS" + "FM"

        __optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.__model.parameters(), 0.05
        )
        __lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            __optimizer, lr_lambda=lambda epoch: 0.8 ** (10 * epoch / __epochs)
        )
        data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _TrainingDataSet(self.__ratings_train_set), __users_batch_size,
            shuffle=True, collate_fn=lambda data: data
        )

        with tqdm.tqdm(range(__epochs)) as epoch_t:
            for current_epoch in epoch_t:
                self.__model.train()
                with tqdm.tqdm(data_loader) as data_loader_t:
                    for batch_users_with_positive_items in data_loader_t:
                        __optimizer.zero_grad()
                        all_users_p_matrix, all_items_q_matrix, h2 = self.__model(
                            self.__all_users_nonzero_indexes_of_features,
                            self.__all_items_nonzero_indexes_of_features
                        )
                        loss: torch.Tensor = self.__compute_loss(
                            h2, 0.5,
                            all_items_q_matrix,
                            all_users_p_matrix,
                            batch_users_with_positive_items
                        )
                        loss_value: float = loss.item()
                        loss.backward()
                        data_loader_t.set_postfix({"loss": str(loss_value)})
                        epoch_t.set_postfix({"loss": str(loss_value)})
                        __optimizer.step()
                __lr_scheduler.step()
                if (current_epoch + 1) % 10 == 0:
                    TorchWeightsUtils.save_state_dict(
                        self.__model.state_dict(),
                        os.path.join(
                            __weights_directory_name,
                            "state-snapshot-%d.pt" % (current_epoch + 1)
                        )
                    )
