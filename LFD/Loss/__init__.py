import torch
import torch.nn.functional


class BPRLoss:
    def __init__(self, batch_size: int, decay: float):
        if type(batch_size) != int or type(decay) != float:
            raise TypeError
        if batch_size <= 0 or decay < 0:
            raise ValueError

        self.__batch_size: int = batch_size
        self.__decay: float = decay

    def __call__(
            self, users_embeddings: torch.Tensor,
            positive_items_embeddings: torch.Tensor,
            negative_items_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the BPR loss
        :param users_embeddings: embeddings of users
        :param positive_items_embeddings: embeddings of positive items
        :param negative_items_embeddings: embeddings of negative items
        :return: loss value
        """
        positive_scores: torch.Tensor = torch.sum(
            torch.mul(users_embeddings, positive_items_embeddings), dim=1
        )
        negative_scores: torch.Tensor = torch.sum(
            torch.mul(users_embeddings, negative_items_embeddings), dim=1
        )
        mf_loss: torch.Tensor = -torch.mean(
            torch.nn.functional.logsigmoid(positive_scores - negative_scores)
        )
        regularization: torch.Tensor = (
            torch.linalg.norm(positive_items_embeddings) +
            torch.linalg.norm(negative_items_embeddings) +
            torch.linalg.norm(users_embeddings)
        ) / 2
        return mf_loss + self.__decay * regularization / self.__batch_size
