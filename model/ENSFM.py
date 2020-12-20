import torch
import typing as _typing


class _EfficientNonSamplingFactorizationMachines(torch.nn.Module):
    def __init__(
            self,
            user_features_number: int,
            item_features_number: int,
            dimension_of_embeddings: int
    ):
        super().__init__()
        self.__user_first_order_feature_interaction_weight = \
            torch.nn.Parameter(torch.randn(user_features_number))
        torch.nn.init.normal_(
            self.__user_first_order_feature_interaction_weight
        )
        self.__item_first_order_feature_interaction_weight = \
            torch.nn.Parameter(torch.randn(item_features_number))
        torch.nn.init.normal_(
            self.__item_first_order_feature_interaction_weight
        )
        self.__global_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(1)
        )
        torch.nn.init.normal_(self.__global_bias)
        self.__h1: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(dimension_of_embeddings)
        )
        torch.nn.init.normal_(self.__h1)
        self.__h2: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(dimension_of_embeddings)
        )
        torch.nn.init.normal_(self.__h2)

    @classmethod
    def __efficient_bi_interaction(
            cls, __features_embeddings: torch.Tensor,
            __features_nonzero_one_hot_indexes: list
    ) -> torch.Tensor:
        """ Implement the efficient f_BI corresponding to Equation(14) in the paper """
        return 0.5 * (
                torch.square(torch.sum(__features_embeddings[__features_nonzero_one_hot_indexes], dim=0)) -
                torch.sum(torch.square(__features_embeddings[__features_nonzero_one_hot_indexes]), dim=0)
        )

    def __build_p_u(
            self, user_features_embeddings: torch.Tensor,
            user_features_nonzero_one_hot_indexes: list,
    ) -> torch.Tensor:
        """ build p_u """
        return torch.cat([
            torch.sum(
                user_features_embeddings[user_features_nonzero_one_hot_indexes], dim=0
            ).reshape(-1),
            torch.matmul(
                self.__h1,
                self.__efficient_bi_interaction(
                    user_features_embeddings,
                    user_features_nonzero_one_hot_indexes
                )
            ) + torch.sum(
                self.__user_first_order_feature_interaction_weight[
                    user_features_nonzero_one_hot_indexes
                ]
            ) + self.__global_bias,
            torch.ones(1)
        ])

    def __build_q_v(
            self, item_features_embeddings: torch.Tensor,
            item_features_nonzero_one_hot_indexes: list
    ) -> torch.Tensor:
        """ build q_v """
        return torch.cat([
            torch.sum(
                item_features_embeddings[item_features_nonzero_one_hot_indexes], dim=0
            ).reshape(-1),
            torch.ones(1),
            torch.matmul(
                self.__h1,
                self.__efficient_bi_interaction(
                    item_features_embeddings,
                    item_features_nonzero_one_hot_indexes
                )
            ) + torch.sum(
                self.__item_first_order_feature_interaction_weight[
                    item_features_nonzero_one_hot_indexes
                ]
            ).reshape(-1)
        ])

    def forward(
            self,
            user_features_embeddings: torch.Tensor,
            item_features_embeddings: torch.Tensor,
            batch_users_nonzero_indexes_of_features: _typing.List[
                _typing.Tuple[int, _typing.List[int]]
            ],
            all_items_nonzero_indexes_of_features: _typing.List[
                _typing.Tuple[int, _typing.List[int]]
            ]
    ) -> _typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        __users_pu_pool: _typing.Dict[int, torch.Tensor] = {}
        __items_qv_pool: _typing.Dict[int, torch.Tensor] = {}
        for __user_id, __current_user_nonzero_indexes_of_features \
                in batch_users_nonzero_indexes_of_features:
            __users_pu_pool[__user_id] = self.__build_p_u(
                user_features_embeddings, __current_user_nonzero_indexes_of_features
            )
        for __item_id, __current_item_nonzero_indexes_of_features \
                in all_items_nonzero_indexes_of_features:
            __items_qv_pool[__item_id] = self.__build_q_v(
                item_features_embeddings, __current_item_nonzero_indexes_of_features
            )
        batch_users_p_matrix: torch.Tensor = torch.stack([
            __users_pu_pool[__user_id]
            for __user_id, __ in batch_users_nonzero_indexes_of_features
        ])
        all_items_q_matrix: torch.Tensor = torch.stack([
            __items_qv_pool[__item_id]
            for __item_id, __ in all_items_nonzero_indexes_of_features
        ])
        return batch_users_p_matrix, all_items_q_matrix, self.__h2

    class Loss(torch.nn.Module):
        def __init__(self, dropout: _typing.Optional[torch.nn.Dropout] = None):
            super().__init__()
            if dropout is None:
                self.__dropout: _typing.Optional[torch.nn.Dropout] = None
            elif isinstance(dropout, torch.nn.Dropout):
                self.__dropout: _typing.Optional[torch.nn.Dropout] = dropout
            else:
                raise TypeError

        def _compute_partial_loss_for_one_interaction(
                self, pu: torch.Tensor, qv: torch.Tensor,
                h2: torch.Tensor, negative_weight: torch.Tensor
        ) -> torch.Tensor:
            __predicted_y: torch.Tensor = torch.matmul(
                torch.cat([
                    h2 if self.__dropout is None else self.__dropout(h2),
                    torch.ones(2)
                ]),
                torch.mul(pu, qv)
            )
            return (1 - negative_weight) * torch.square(__predicted_y) - 2 * __predicted_y

        def forward(
                self, h2: torch.Tensor,
                negative_weight: float,
                all_items_q_matrix: torch.Tensor,
                batch_users_p_matrix: torch.Tensor,
                batch_users_with_positive_items: _typing.List[
                    _typing.Tuple[int, _typing.List[int]]
                ]
        ) -> torch.Tensor:
            if h2.size() != (batch_users_p_matrix.size(1) - 2,):
                raise ValueError
            if all_items_q_matrix.size(1) != batch_users_p_matrix.size(1):
                raise ValueError
            if batch_users_p_matrix.size(0) != len(batch_users_with_positive_items):
                raise ValueError
            ''' Compute the first term of loss '''
            loss: torch.Tensor = torch.zeros(1)
            for __user_index in range(len(batch_users_with_positive_items)):
                for __item_index in batch_users_with_positive_items[__user_index][1]:
                    loss += self._compute_partial_loss_for_one_interaction(
                        batch_users_p_matrix[__user_index],
                        all_items_q_matrix[__item_index],
                        h2, torch.tensor(negative_weight)
                    )
            ''' Compute the second term of loss '''
            __intermediate_matrix: torch.Tensor = torch.mul(
                torch.mm(
                    torch.cat([h2, torch.ones(2)]).reshape((-1, 1)),
                    torch.cat([h2, torch.ones(2)]).reshape((1, -1))
                ),
                torch.mul(
                    torch.mm(batch_users_p_matrix.t(), batch_users_p_matrix),
                    torch.mm(all_items_q_matrix.t(), all_items_q_matrix)
                )
            )
            return loss + negative_weight * __intermediate_matrix.sum()


class ENSFMWithEmbeddings(torch.nn.Module):
    def __init__(
            self,
            user_features_number: int,
            item_features_number: int,
            dimension_of_embeddings: int
    ):
        super().__init__()
        self.__efficient_non_sampling_factorization_machines = \
            _EfficientNonSamplingFactorizationMachines(
                user_features_number,
                item_features_number,
                dimension_of_embeddings
            )
        self.__user_features_embedding: torch.nn.Embedding = \
            torch.nn.Embedding(user_features_number, dimension_of_embeddings)
        self.__item_features_embedding: torch.nn.Embedding = \
            torch.nn.Embedding(item_features_number, dimension_of_embeddings)

    def forward(
            self,
            batch_users_nonzero_indexes_of_features: _typing.List[
                _typing.Tuple[int, _typing.List[int]]
            ],
            all_items_nonzero_indexes_of_features: _typing.List[
                _typing.Tuple[int, _typing.List[int]]
            ]
    ) -> _typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.__efficient_non_sampling_factorization_machines(
            self.__user_features_embedding.weight,
            self.__item_features_embedding.weight,
            batch_users_nonzero_indexes_of_features,
            all_items_nonzero_indexes_of_features
        )

    class Loss(_EfficientNonSamplingFactorizationMachines.Loss):
        ...
