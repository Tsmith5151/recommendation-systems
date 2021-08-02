import annoy
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def cosine_similarity(user: int, user_weights: np.ndarray, top_k: int):
    """Helper function to compute pairwise distance similarity
    to obtain top-K similar users
    """
    distance = 1 - pairwise_distances(user_weights, metric="cosine")
    ranking = np.argsort(distance, axis=0)[::-1][1 : top_k + 1].T
    return ranking[user - 1]


class ApproximateTopRelated:
    """Approximated Nearest Neighbors"""

    def __init__(self, factors, treecount=20):
        index = annoy.AnnoyIndex(factors.shape[1], "angular")
        for i, row in enumerate(factors):
            index.add_item(i, row)
        index.build(treecount)
        self.index = index

    def get_related(self, idx, topk=10):
        neighbors = self.index.get_nns_by_item(idx, topk)
        return sorted(
            ((other, 1 - self.index.get_distance(idx, other)) for other in neighbors),
            key=lambda x: -x[1],
        )
