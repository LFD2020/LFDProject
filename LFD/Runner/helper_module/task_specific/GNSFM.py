import torch

import typing as _typing

from LFD.Models.NGCF import NeuralGraphCollaborativeFiltering
from LFD.Models.ENSFM import EfficientNonSamplingFactorizationMachines


class GraphBasedNonSamplingFactorizationMachines(torch.nn.Module):
    """ Graph-based Non-Sampling Factorization Machines for MovieLens-1M """

    def __init__(
            self,
            number_of_users: int, number_of_movies: int
    ):
        super().__init__()
        self.__number_of_users: int = number_of_users
        self.__number_of_movies: int = number_of_movies
        self.__number_of_genres: int = 18
        self.__number_of_age_groups: int = 7
        self.__number_of_occupations: int = 21

        self.__embedding_of_user_properties: torch.nn.Embedding = torch.nn.Embedding(
            2 + self.__number_of_age_groups + self.__number_of_occupations, 256
        )
        self.__embedding_of_movie_properties: torch.nn.Embedding = torch.nn.Embedding(
            self.__number_of_genres, 256
        )
        torch.nn.init.xavier_normal_(self.__embedding_of_user_properties.weight)
        torch.nn.init.xavier_normal_(self.__embedding_of_movie_properties.weight)

        self.__neural_graph_collaborative_filtering: NeuralGraphCollaborativeFiltering = \
            NeuralGraphCollaborativeFiltering(
                self.__number_of_users, self.__number_of_movies
            )
        self.__efficient_non_sampling_factorization_machines: EfficientNonSamplingFactorizationMachines = \
            EfficientNonSamplingFactorizationMachines(
                self.__number_of_users + 2 + self.__number_of_age_groups + self.__number_of_occupations,
                self.__number_of_movies + self.__number_of_genres,
                256
            )

    def forward(
            self, l_matrix: torch.sparse.Tensor,
            batch_users_nonzero_indexes_of_features: _typing.List[
                _typing.Tuple[int, _typing.List[int]]
            ],
            all_items_nonzero_indexes_of_features: _typing.List[
                _typing.Tuple[int, _typing.List[int]]
            ]
    ) -> _typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings_of_user_ids, embeddings_of_movie_ids = \
            self.__neural_graph_collaborative_filtering(l_matrix)
        return self.__efficient_non_sampling_factorization_machines(
            torch.cat([embeddings_of_user_ids, self.__embedding_of_user_properties.weight]),
            torch.cat([embeddings_of_movie_ids, self.__embedding_of_movie_properties.weight]),
            batch_users_nonzero_indexes_of_features, all_items_nonzero_indexes_of_features
        )

    class Loss(EfficientNonSamplingFactorizationMachines.Loss):
        ...
