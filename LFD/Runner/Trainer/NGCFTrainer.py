from LFD.DataSource.MovieLens.datasets.MovieLens1M2003 import MovieLens1M2003
from LFD.DataSource.MovieLens.preprocessor.ratings import *
from LFD.DataSource.MovieLens.sampler import MovieLensSampler
from LFD.Models.NGCF import NeuralGraphCollaborativeFiltering
from LFD.Loss import BPRLoss
from LFD.weights import TorchWeightsUtils

import pandas as pd
import tqdm


class NeuralGraphCollaborativeFilteringTrainer:
    def __init__(self, gpu_available: bool = torch.cuda.is_available()):
        self._number_of_epochs: int = 500
        self._batch_size: int = 4096

        movie_lens_1m_2003: MovieLens1M2003 = MovieLens1M2003()
        self._number_of_users: int = movie_lens_1m_2003.number_of_users
        self._number_of_items: int = movie_lens_1m_2003.number_of_movies
        self._train_set: pd.DataFrame = movie_lens_1m_2003.load_train_set()
        self._sampler: MovieLensSampler = MovieLensSampler(
            self._train_set, self._number_of_users, self._number_of_items
        )
        self._matrices = MatricesGenerator(
            self._train_set, self._number_of_users, self._number_of_items
        )
        del movie_lens_1m_2003

        self._model: NeuralGraphCollaborativeFiltering = NeuralGraphCollaborativeFiltering(
            self._number_of_users, self._number_of_items
        )
        if gpu_available:
            self._model: NeuralGraphCollaborativeFiltering = self._model.cuda()
        self._bpr_loss: BPRLoss = BPRLoss(self._batch_size, decay=1e-5)
        self._optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self._model.parameters(), lr=0.001
        )
        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lr_lambda=lambda epoch: 0.96 ** (10.0 * epoch / self._batch_size)
        )

        self._weights_directory: str = "N" + "GCF"

    def train(self):
        with tqdm.tqdm(range(self._number_of_epochs)) as t_epoch:
            for epoch in t_epoch:
                self._model.train()
                with tqdm.tqdm(range(self._sampler.number_of_ratings // self._batch_size + 1)) as t_batch:
                    for _ in t_batch:
                        self._optimizer.zero_grad()
                        all_users_embeddings, all_items_embeddings = self._model(self._matrices.normalized_l)

                        positive_samples, negative_samples = self._sampler.sample(self._batch_size)

                        sampled_users_id_list: list = positive_samples['userId'].tolist()
                        sampled_positive_movies: list = positive_samples['movieId'].tolist()
                        sampled_negative_movies: list = negative_samples['movieId'].tolist()

                        assert len(sampled_users_id_list) == len(sampled_positive_movies) == len(sampled_negative_movies)

                        loss: torch.Tensor = self._bpr_loss(
                            all_users_embeddings[sampled_users_id_list],
                            all_items_embeddings[sampled_positive_movies],
                            all_items_embeddings[sampled_negative_movies]
                        )
                        t_batch.set_postfix({"loss": str(loss.item())})

                        loss.backward()
                        self._optimizer.step()
                self._lr_scheduler.step()
