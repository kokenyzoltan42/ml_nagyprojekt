import numpy as np


class TSNE:
    def __init__(self, data, ydim=2, num_iterations=100, learning_rate=500, perp=30):
        self.data = data
        self.ydim = ydim
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.perp = perp

    def run(self) -> np.array:
        """
        Lefuttatja az algoritmust,
        A felépítése nagyjából azonos a pszeudokódban lévővel
        :return: Alacsony dimenziós adatreprezentáció
        """
        # Adatpontok számának meghatározása
        N = self.data.shape[0]
        # p_{ij}-k kiszámolása
        P = self._p_joint()

        map_points = []
        # (0, 10^{-4}) paraméterű normális eloszlás szerint generálja le
        # véletlenszerűen a sík (/ adott dimenzió) pontjait
        new_mappoints = np.random.normal(loc=0.0, scale=1e-4, size=(N, self.ydim))
        # Mivel az iterációs képletben már használjuk a 2-vel korábbi iteráció eredményeit,
        # ezért még az iteráció kezdete előtt kétszer adjuk meg a kezdő pontokat
        map_points.append(new_mappoints)
        map_points.append(new_mappoints)

        for t in range(self.num_iterations):
            Q = self._q_i_j(map_points[-1])
            if t < 50:
                # Gradiens kiszámolása
                # Az optimalizálás kezdetekor a kiszámított P-ket 4-el beszorozzuk
                # Az eredeti cikk is csak az első 50 iterációnál használta
                grad = self._gradient(P=4 * P, Q=Q, y=map_points[-1])
            else:
                grad = self._gradient(P=P, Q=Q, y=map_points[-1])
            # Alacsony dimenziós pontok t-edik iterációjának meghatározása:
            new_mappoints = map_points[-1] - self.learning_rate * grad + \
                            self._momentum(t) * (map_points[-1] - map_points[-2])
            map_points.append(new_mappoints)

            # "Optimalizálás"-hoz van, lehet kiszedhetjük (?)
            if t % 10 == 0:
                Q = np.maximum(Q, 1e-12)
        return new_mappoints

    @staticmethod
    def _pairwise_distances(data: np.array) -> np.array:
        """
        :param data: pontok koordinátasora
        :return: Az pontok páronkénti euklideszi távolságának meghatározása
        """
        return np.sum((data[None, :] - data[:, None]) ** 2, axis=2)

    @staticmethod
    def _p_i_j(dists: np.array, sigma: np.array) -> np.array:
        """
        Páronkénti szomszédságot / hasonlóságot mérő
        feltételes valószínűségek meghatározása (p_{i|j})
        :param dists: páronkénti euklideszi távolságok
        :param sigma: Pontokhoz tartozó szórások
        :return: Annak a valószínűsége, hogy az x_j szomszédjának választja az x_i-ket, vektorba rendezve 
        """
        e = np.exp(-dists / (2 * np.square(sigma.reshape((-1, 1)))))
        # Megszabjuk, hogy p_{i|i} nullával legyen egyenlő
        np.fill_diagonal(e, val=0.)
        # 0-val való osztás elkerülése  érdekében
        e += 1e-8
        return e / e.sum(axis=1).reshape([-1, 1])

    @staticmethod
    def _p_j_i(dists: np.array, sigma: np.array) -> np.array:
        """
        Páronkénti szomszédságot / hasonlóságot mérő
        feltételes valószínűségek meghatározása (p_{j|i})
        :param dists: páronkénti euklideszi távolságok
        :param sigma: Pontokhoz tartozó szórások
        :return: Annak a valószínűsége, hogy az x_i szomszédjának választja x_j-ket, vektorba rendezve
        """
        e = np.exp(-dists / (2 * np.square(sigma.reshape((-1, 1)))))
        # Megszabjuk, hogy p_{j|j} nullával legyen egyenlő
        np.fill_diagonal(e, val=0.)
        # 0-val való osztás elkerülése  érdekében
        e += 1e-8
        return e / e.sum(axis=0).reshape([-1, 1])

    def _p_joint(self) -> np.array:
        """
        :return: szimmetrikus valószínűségek kiszámolása
        """
        N = self.data.shape[0]
        dists = self._pairwise_distances(data=self.data)
        sigmas = self._find_sigmas(dists=dists, perplexity=self.perp)
        p_cond_1 = self._p_i_j(dists=dists, sigma=sigmas)
        p_cond_2 = self._p_j_i(dists=dists, sigma=sigmas)
        return (p_cond_1 + p_cond_2) / (2.*N)

    @staticmethod
    def _perp(p_cond: np.array) -> float:
        # Perp kiszámolása (3) szerint
        entropy = -np.sum(p_cond * np.log2(p_cond), axis=1)
        return 2 ** entropy

    def _find_sigmas(self, dists: np.array, perplexity: int) -> np.array:
        """
        Az adatpontohoz tartozó szórás, szigmák keresése
        :param dists: adatpontok távolsága
        :param perplexity: perplexity
        :return: adatpontokhoz tartozó szórások
        """
        found_sigmas = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            func = lambda sig: self._perp(self._p_i_j(dists=dists[i:i + 1, :],
                                                      sigma=np.array([sig])))
            found_sigmas[i] = self._binary_search(func=func, goal=perplexity)
        return found_sigmas

    @staticmethod
    def _binary_search(func, goal: int, tol=1e-10, max_iters: int = 1000, lowb: int = 1e-20,
                       uppb: int = 10000) -> float:
        """
        Bináris keresés - intervallum felező módszer segítségével keressük a szórást,
        vagyis normális eloszlásbeli szigmákat
        :param func: Perp függvény, amit meg akarunk oldani
        :param goal: az a szigma, ami kielégíti a fenti egyenletet
        :param tol: tolerancia nagysága
        :param max_iters: megengedett maximális iteráció szám
        :param lowb: alsó határ
        :param uppb: felső határ
        :return: a func egyenletet kielégító szigma
        """
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
        """
        :param y: alacsonyabb dimenziójú pontok
        :return: térképpontok szomszédságainak / hasonlóságainak vektorai
        """
        # alacsonyabb dimenziós pontok (térképpontok) közötti euklideszi távolság
        dists = self._pairwise_distances(y)
        nom = 1 / (1 + dists)
        # q_{ii}-ket 0-val tesszük egyenlővé
        np.fill_diagonal(nom, val=0.)
        return nom / np.sum(np.sum(nom))

    def _gradient(self, P: np.array, Q: np.array, y: np.array) -> np.array:
        """
        A költségfüggvény gradiensének kiszámolása
        :param P: Magasabb dimenziós pontok szomszédságának vektorai
        :param Q: alacsony dimenziós pontok szomszédsági vektorai
        :param y: térképpontok
        :return: gradiensvektor
        """
        pq_diff = P - Q
        y_diff = np.expand_dims(y, axis=1) - np.expand_dims(y, axis=0)
        dists = self._pairwise_distances(data=y)
        aux = 1 / (1 + dists)
        result = 4 * (np.expand_dims(pq_diff, axis=2) * y_diff * np.expand_dims(aux, axis=2)).sum(axis=1)
        return result

    @staticmethod
    def _momentum(t: int) -> float:
        """
        :param t: t-edik iteráció
        :return: \alpha momentum a t-edik iterációnál
        """
        return 0.5 if t < 250 else 0.8

# TODO:
#  lehet nem kell a függvények elé a _
