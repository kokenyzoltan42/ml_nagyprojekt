import numpy as np


class TSNE:
    def __init__(self, data, ydim=2, num_iterations=100, learning_rate=500, perp=30):
        self.data = data
        self.ydim = ydim
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.perp = perp

    def run(self) -> np.array:
        N = self.data.shape[0]
        P = self._p_joint()

        map_points = []
        new_mappoints = np.random.normal(loc=0.0, scale=1e-4, size=(N, self.ydim))
        map_points.append(new_mappoints)
        map_points.append(new_mappoints)

        for t in range(self.num_iterations):
            Q = self._q_i_j(map_points[-1])
            if t < 50:
                # Az optimalizálás kezdetekor a kiszámított P-ket 4-el beszorozzuk
                # Az eredeti cikk is csak az első 50 iterációnál használta
                grad = self._gradient(P=4 * P, Q=Q, y=map_points[-1])
            else:
                grad = self._gradient(P=P, Q=Q, y=map_points[-1])
            new_mappoints = map_points[-1] - self.learning_rate * grad + \
                            self._momentum(t) * (map_points[-1] - map_points[-2])
            map_points.append(new_mappoints)

            if t % 10 == 0:
                Q = np.maximum(Q, 1e-12)
        return new_mappoints

    @staticmethod
    def _pairwise_distances(data: np.array) -> np.array:
        return np.sum((data[None, :] - data[:, None]) ** 2, axis=2)

    @staticmethod
    def _p_i_j(dists: np.array, sigma: np.array) -> np.array:
        e = np.exp(-dists / (2 * np.square(sigma.reshape((-1, 1)))))
        np.fill_diagonal(e, val=0.)
        e += 1e-8
        return e / e.sum(axis=1).reshape([-1, 1])

    @staticmethod
    def _p_j_i(dists: np.array, sigma: np.array) -> np.array:
        e = np.exp(-dists / (2 * np.square(sigma.reshape((-1, 1)))))
        np.fill_diagonal(e, val=0.)
        e += 1e-8
        return e / e.sum(axis=0).reshape([-1, 1])

    def _p_joint(self) -> np.array:
        N = self.data.shape[0]
        dists = self._pairwise_distances(data=self.data)
        sigmas = self._find_sigmas(dists=dists, perplexity=self.perp)
        p_cond_1 = self._p_i_j(dists=dists, sigma=sigmas)
        p_cond_2 = self._p_j_i(dists=dists, sigma=sigmas)
        return (p_cond_1 + p_cond_2) / (2. * N)

    @staticmethod
    def _perp(p_cond: np.array) -> float:
        entropy = -np.sum(p_cond * np.log2(p_cond), axis=1)
        return 2 ** entropy

    def _find_sigmas(self, dists: np.array, perplexity: int) -> np.array:
        found_sigmas = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            func = lambda sig: self._perp(self._p_i_j(dists=dists[i:i + 1, :],
                                                      sigma=np.array([sig])))
            found_sigmas[i] = self._binary_search(func=func, goal=perplexity)
        return found_sigmas

    @staticmethod
    def _binary_search(func, goal: int, tol=1e-10, max_iters: int = 1000, lowb: int = 1e-20,
                       uppb: int = 10000) -> float:
        guess = 0
        for _ in range(max_iters):
            guess = (uppb + lowb) / 2.
            val = func(guess)

            if val > goal:
                uppb = guess
            else:
                lowb = guess

            if np.abs(val - goal) <= tol:
                return guess
        return guess

    def _q_i_j(self, y: np.array) -> np.array:
        dists = self._pairwise_distances(y)
        nom = 1 / (1 + dists)
        np.fill_diagonal(nom, val=0.)
        return nom / np.sum(np.sum(nom))

    def _gradient(self, P: np.array, Q: np.array, y: np.array) -> np.array:
        # (n, no_dims) = y.shape - Nem használjuk
        pq_diff = P - Q
        y_diff = np.expand_dims(y, axis=1) - np.expand_dims(y, axis=0)
        dists = self._pairwise_distances(data=y)
        aux = 1 / (1 + dists)
        result = 4 * (np.expand_dims(pq_diff, axis=2) * y_diff * np.expand_dims(aux, axis=2)).sum(axis=1)
        return result

    @staticmethod
    def _momentum(t: int) -> float:
        return 0.5 if t < 250 else 0.8

# TODO: docstring-ek hozzáadása,
#  és ellenőrzés
#  kell az early exagguration?
#  lehet nem kell a függvények elé a _
