import torch
import typing as _typing


class EfficientNonSamplingFactorizationMachines(torch.nn.Module):
    def __init__(
            self,
            user_features_number: int,
            item_features_number: int,
            dimension_of_embeddings: int,
            negative_weight: int = 0.5
    ):
        super().__init__()
        self.__user_first_order_feature_interaction_weight = \
            torch.nn.Parameter(torch.rand(user_features_number))
        self.__item_first_order_feature_interaction_weight = \
            torch.nn.Parameter(torch.rand(item_features_number))
        self.__global_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(1)
        )

        self.__h1: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(dimension_of_embeddings)
        )
        self.__h2: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(dimension_of_embeddings)
        )

        self.__negative_weight = negative_weight

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

    def _build_p_u(
            self, user_features_embeddings: torch.Tensor,
            user_features_nonzero_one_hot_indexes: list,
    ) -> torch.Tensor:
        """ build p_u """

        ''' Generate p_{u,d} '''
        p_ud: torch.Tensor = torch.sum(
            user_features_embeddings[user_features_nonzero_one_hot_indexes], dim=0
        )
        assert len(p_ud.size()) == 1
        ''' Generate p_u '''
        return torch.cat([
            p_ud,
            torch.matmul(
                self.__h1.t(),
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

    def _build_q_v(
            self, item_features_embeddings: torch.Tensor,
            item_features_nonzero_one_hot_indexes: list
    ) -> torch.Tensor:
        """ build q_v """

        ''' Generate q_{v,d} '''
        q_vd: torch.Tensor = torch.sum(
            item_features_embeddings[item_features_nonzero_one_hot_indexes], dim=0
        )
        assert len(q_vd.size()) == 1
        ''' Generate q_v '''
        return torch.cat([
            q_vd,
            torch.ones(1),
            torch.matmul(
                self.__h1.t(),
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

    def __compute_loss_with_multiprocessing(
            self,
            user_features_embeddings: torch.Tensor,
            item_features_embeddings: torch.Tensor,
            batch_users_with_positive_items: _typing.Dict[
                int, _typing.Tuple[_typing.List[int], _typing.List[int]]
            ],
            all_items_features_nonzero_one_hot_indexes: _typing.Dict[
                int, _typing.List[int]
            ]
    ) -> torch.Tensor:
        import multiprocessing

        __users_p_u_pool: _typing.Dict[int, torch.Tensor] = {}
        __items_q_v_pool: _typing.Dict[int, torch.Tensor] = {}
        with multiprocessing.Pool() as pool:
            def __build_p_u_for_one_user(
                    __user_id: int,
                    __user_features_nonzero_one_hot_indexes: list
            ) -> None:
                __users_p_u_pool[__user_id] = self._build_p_u(
                    user_features_embeddings,
                    __user_features_nonzero_one_hot_indexes
                )

            def __build_q_v_for_one_item(
                    __item_id: int,
                    __item_features_nonzero_one_hot_indexes: list
            ) -> None:
                __items_q_v_pool[__item_id] = self._build_q_v(
                    item_features_embeddings,
                    __item_features_nonzero_one_hot_indexes
                )

            for __user_id, __data_of_current_user \
                    in batch_users_with_positive_items.items():
                pool.apply_async(
                    __build_p_u_for_one_user,
                    (__user_id, __data_of_current_user[0])
                )
            for __item_id, __current_item_features_nonzero_one_hot_indexes \
                    in all_items_features_nonzero_one_hot_indexes.items():
                pool.apply_async(
                    __build_q_v_for_one_item,
                    (__item_id, __current_item_features_nonzero_one_hot_indexes)
                )
            pool.close()
            pool.join()

        ''' Compute the last term of loss '''
        __all_users_p_matrix: torch.Tensor = torch.stack(
            [__current_p_u for __current_p_u in __users_p_u_pool.values()]
        )
        __all_items_q_matrix: torch.Tensor = torch.stack(
            [__current_q_v for __current_q_v in __items_q_v_pool.values()]
        )
        __intermediate_matrix: torch.Tensor = torch.mul(
            torch.mm(
                torch.cat([self.__h2, torch.ones(2)]).reshape((-1, 1)),
                torch.cat([self.__h2, torch.ones(2)]).reshape((1, -1))
            ),
            torch.mul(
                torch.mm(__all_users_p_matrix.t(), __all_users_p_matrix),
                torch.mm(__all_items_q_matrix.t(), __all_items_q_matrix)
            )
        )
        loss: torch.Tensor = self.__negative_weight * __intermediate_matrix.sum()
        assert loss.size() == (1,)

        ''' Compute the First term of loss '''
        with multiprocessing.Pool() as pool:
            def __compute_partial_loss_for_one_record(
                    __pu: torch.Tensor, __qv: torch.Tensor
            ) -> torch.Tensor:
                __predicted_y: torch.Tensor = torch.matmul(
                    torch.cat([self.__h2, torch.ones(2)]),
                    torch.mul(__pu, __qv)
                )
                assert __predicted_y.size() == (1,)
                return (
                        (1 - self.__negative_weight) * torch.square(__predicted_y) -
                        2 * __predicted_y
                )

            __async_results: _typing.List[
                multiprocessing.pool.AsyncResult
            ] = []
            for __user_id, __data_of_current_user \
                    in batch_users_with_positive_items.items():
                for __item_id in __data_of_current_user[1]:
                    __async_results.append(pool.apply_async(
                        __compute_partial_loss_for_one_record,
                        (__users_p_u_pool[__user_id], __items_q_v_pool[__item_id])
                    ))
            pool.close()
            pool.join()

            loss += torch.cat(
                [async_result.get() for async_result in __async_results]
            ).sum()
        assert loss.size() == (1,)
        return loss

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
            __users_pu_pool[__user_id] = self._build_p_u(
                user_features_embeddings, __current_user_nonzero_indexes_of_features
            )
        for __item_id, __current_item_nonzero_indexes_of_features \
                in all_items_nonzero_indexes_of_features:
            __items_qv_pool[__item_id] = self._build_q_v(
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
        @classmethod
        def _compute_partial_loss_for_one_interaction(
                cls, pu: torch.Tensor, qv: torch.Tensor,
                h2: torch.Tensor, negative_weight: torch.Tensor
        ) -> torch.Tensor:
            __predicted_y: torch.Tensor = torch.matmul(
                torch.cat([h2, torch.ones(2)]), torch.mul(pu, qv)
            )
            assert __predicted_y.size() == (1,)
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
            import multiprocessing
            with multiprocessing.Pool() as pool:
                __async_results: _typing.List[multiprocessing.pool.AsyncResult] = []
                for __user_index in range(len(batch_users_with_positive_items)):
                    for __item_index in batch_users_with_positive_items[__user_index][1]:
                        __async_results.append(pool.apply_async(
                            self._compute_partial_loss_for_one_interaction,
                            (
                                batch_users_p_matrix[__user_index],
                                all_items_q_matrix[__item_index],
                                h2, negative_weight
                            )
                        ))
                pool.close()
                pool.join()
                loss: torch.Tensor = \
                    torch.cat([async_result.get() for async_result in __async_results]).sum()
            assert loss.size() == (1,)
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
