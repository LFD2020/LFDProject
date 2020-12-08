import typing as _typing
import torch
import torch.sparse
import torch.nn.functional
import torch.nn.init


class NeuralGraphCollaborativeFiltering(torch.nn.Module):
    class _PropagationLayer(torch.nn.Module):
        def __init__(
                self,
                last_feature_dimension: int = 64,
                next_feature_dimension: int = 64,
                dropout_probability: float = 0.1
        ):
            super().__init__()
            self.__linear_transforms: torch.nn.ModuleList = torch.nn.ModuleList(
                [torch.nn.Linear(last_feature_dimension, next_feature_dimension) for _ in range(3)]
            )
            self.__dropout_probability: float = dropout_probability
            self.__dropout: torch.nn.Dropout = torch.nn.Dropout(dropout_probability)

        def forward(self, l: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
            term1: torch.Tensor = torch.sparse.mm(l, feature)
            term2: torch.Tensor = torch.mul(term1, feature)
            intermediate: torch.Tensor = torch.nn.functional.leaky_relu(
                self.__linear_transforms[0](term1) +
                self.__linear_transforms[1](term2) +
                self.__linear_transforms[2](feature)
            )
            return torch.nn.functional.normalize(self.__dropout(intermediate))

    def __init__(
            self, n_users: int, n_items: int, embedding_dimension: int = 64,
            propagation_dimensions: _typing.List[int] = (64, 64, 64),
            dropout_probabilities: _typing.List[float] = (0.1, 0.1, 0.1)
    ):
        super().__init__()
        self.__n_users: int = n_users
        self.__n_items: int = n_items
        self.__user_embedding: torch.nn.Embedding = torch.nn.Embedding(n_users, embedding_dimension)
        self.__item_embedding: torch.nn.Embedding = torch.nn.Embedding(n_items, embedding_dimension)
        self.__propagation: torch.nn.Sequential = torch.nn.Sequential()

        __temp_embedding_dimensions: list = [embedding_dimension] + list(propagation_dimensions)
        self.__propagation_layers = torch.nn.ModuleList(
            [
                self._PropagationLayer(
                    __temp_embedding_dimensions[i - 1], __temp_embedding_dimensions[i],
                    dropout_probability=dropout_probabilities[i - 1]
                ) for i in range(1, len(__temp_embedding_dimensions))
            ]
        )

        torch.nn.init.xavier_normal_(self.__user_embedding.weight)
        torch.nn.init.xavier_normal_(self.__item_embedding.weight)

    def forward(self, l: torch.sparse.Tensor) -> _typing.Tuple[torch.Tensor, torch.Tensor]:
        assert l.shape == (self.__n_users + self.__n_items, self.__n_users + self.__n_items)
        embeddings: torch.Tensor = torch.cat(
            (self.__user_embedding.weight, self.__item_embedding.weight), dim=0
        )
        all_embeddings = [embeddings]
        for i in range(len(self.__propagation_layers)):
            embeddings: torch.Tensor = self.__propagation_layers[i](l, embeddings)
            all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=1)
        user_embeddings, item_embeddings = torch.split(
            all_embeddings, [self.__n_users, self.__n_items], dim=0
        )
        return user_embeddings, item_embeddings
