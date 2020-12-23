import os
import pickle
import typing as _typing

import pandas as pd
import torch.utils.data
import tqdm

from LFD.Runner.helper_module.task_specific.GNSFM import GraphBasedNonSamplingFactorizationMachines
from LFD.DataSource.MovieLens.preprocessor.ratings import MatricesGenerator
from LFD.weights import TorchWeightsUtils


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


class GraphBasedNonSamplingFactorizationMachinesTrainer:
    """ Trainer for Graph-based Non-Sampling Factorization Machines on MovieLens-1M """

    def __init__(self, data_directory_path: str):
        """
        :param data_directory_path: path to data
        """
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

        self._model: GraphBasedNonSamplingFactorizationMachines = \
            GraphBasedNonSamplingFactorizationMachines(
                len(self.__preprocessed_users_data),
                len(self.__preprocessed_movies_data)
            )
        '''
        load train data set, according to the hybrid model,
        train data set required for both training and testing
        '''
        self.__ratings_train_set: pd.DataFrame = pd.read_csv(
            os.path.join(self.__data_directory_path, 'train.csv')
        )

        self.__normalized_l_matrix: torch.sparse.Tensor = \
            MatricesGenerator(
                self.__ratings_train_set,
                len(self.__preprocessed_users_data),
                len(self.__preprocessed_movies_data)
            ).normalized_l

        ''' load all items nonzero indexes of features '''
        number_of_movies: int = len(self.__preprocessed_movies_data)
        self.__all_items_nonzero_indexes_of_features: _typing.List[
            _typing.Tuple[int, _typing.List[int]]
        ] = []
        for _movie_index in range(number_of_movies):
            _movie_id = self.__preprocessed_movies_data.iat[_movie_index, 3]
            _movie_genres = self.__preprocessed_movies_data.iat[_movie_index, 4]
            _movie_genres += number_of_movies
            self.__all_items_nonzero_indexes_of_features.append(
                (_movie_id, [_movie_id] + _movie_genres.tolist())
            )
        ''' load all users nonzero indexes of features '''
        number_of_users: int = len(self.__preprocessed_users_data)
        self.__all_users_nonzero_indexes_of_features: _typing.List[
            _typing.Tuple[int, _typing.List[int]]
        ] = []
        for _user_index in range(number_of_users):
            _user_id = self.__preprocessed_users_data.iat[_user_index, 0]
            _gender = self.__preprocessed_users_data.iat[_user_index, 1]
            _gender += number_of_users
            _age = self.__preprocessed_users_data.iat[_user_index, 2]
            _age += number_of_users + 2
            _occupation = self.__preprocessed_users_data.iat[_user_index, 3]
            _occupation += number_of_users + 2 + 7
            self.__all_users_nonzero_indexes_of_features.append(
                (_user_id, [_user_id, _gender, _age, _occupation])
            )

        self.__weights_directory_name: str = "G" + "NS" + "FM"

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
        loss_function = GraphBasedNonSamplingFactorizationMachines.Loss()
        return loss_function(
            h2, negative_weight,
            all_items_q_matrix,
            batch_users_p_matrix,
            batch_users_with_positive_items
        )

    def train(self, initial_weights_filename: _typing.Optional[str] = None):
        if (
                initial_weights_filename is not None and
                type(initial_weights_filename) != str
        ):
            raise TypeError

        if initial_weights_filename is not None:
            if type(initial_weights_filename) != str:
                raise TypeError
            if not os.path.exists(os.path.join(
                    self.__weights_directory_name, initial_weights_filename
            )):
                raise FileNotFoundError
            if not os.path.isdir(os.path.join(
                    self.__weights_directory_name, initial_weights_filename
            )):
                raise IsADirectoryError
            assert os.path.isfile(os.path.join(
                self.__weights_directory_name, initial_weights_filename
            ))
            self._model.load_state_dict(TorchWeightsUtils.load_state_dict(
                os.path.join(self.__weights_directory_name, initial_weights_filename)
            ))

        __epochs: int = 1000
        __users_batch_size: int = 512

        ''' Set the optimizer and lr_scheduler '''
        _optimizer: torch.optim.Optimizer = torch.optim.Adam(self._model.parameters(), 0.05)
        _lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            _optimizer, lr_lambda=lambda epoch: 0.9 ** (10.0 * epoch / __epochs)
        )

        data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _TrainingDataSet(self.__ratings_train_set), __users_batch_size,
            shuffle=True, collate_fn=lambda data: data
        )

        with tqdm.tqdm(range(__epochs)) as epoch_t:
            for current_epoch in epoch_t:
                self._model.train()
                with tqdm.tqdm(data_loader) as data_loader_t:
                    for batch_users_with_positive_items in data_loader_t:
                        _optimizer.zero_grad()
                        batch_users_p_matrix, all_items_q_matrix, h2 = self._model(
                            self.__normalized_l_matrix,
                            self.__all_users_nonzero_indexes_of_features,
                            self.__all_items_nonzero_indexes_of_features
                        )
                        loss: torch.Tensor = self.__compute_loss(
                            h2, 0.5,
                            all_items_q_matrix,
                            batch_users_p_matrix,
                            batch_users_with_positive_items
                        )
                        loss_value: float = loss.item()
                        loss.backward()
                        epoch_t.set_postfix({"loss": str(loss_value)})
                        data_loader_t.set_postfix({"loss": str(loss_value)})
                        _optimizer.step()
                _lr_scheduler.step()

                if (current_epoch + 1) % 10 == 0:
                    TorchWeightsUtils.save_state_dict(
                        self._model.state_dict(),
                        os.path.join(
                            self.__weights_directory_name,
                            "state-snapshot-%d.pt" % (current_epoch + 1)
                        )
                    )
