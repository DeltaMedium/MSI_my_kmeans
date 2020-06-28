from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from tabulate import tabulate
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from k_means_off import *


clfs = {
    'K_Means_my': my_kmeans_msi(),
    'OPTICS': OPTICS(),
    'MeanShift': MeanShift(),
    'DBscan': DBSCAN()
}

metrics = ['---->   METRYKA 1 : adjusted_rand_score    <----',
           '---->   METRYKA 2 : v_measure_score        <----',
           '---->   METRYKA 3 : homogeneity_score      <----',
           '---->   METRYKA 4 : completeness_score     <----']


"""datasets = ['australian', 'balance',
            'cryotherapy', 'diabetes',
            'digitXXXX', 'german',
            'heart', 'liver',
            'soybean', 'waveform']"""
# Z uwagi na ograniczenia komputera nie byłem w stanie zrealizować pobrania danych za jednym razem i musiałem to "rozbić" na kilka uruchomień programu
datasets = ['digit']


#datasets = ['australian']

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    print(datasets[data_id])

    for round_timer in range(4):
        if (round_timer == 0):
            print(metrics[round_timer])
        if (round_timer == 1):
            print(metrics[round_timer])
        if (round_timer == 2):
            print(metrics[round_timer])
        if (round_timer == 3):
            print(metrics[round_timer])
        n_datasets = len(datasets)
        n_splits = 5
        n_repeats = 2
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
        scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                clf = clone(clfs[clf_name])
                clf.fit(X[train], y[train])
                y_pred = clf.fit_predict(X[test])
                if (round_timer == 0):
                    scores[clf_id, data_id, fold_id] = adjusted_rand_score(y[test], y_pred)
                if (round_timer == 1):
                    scores[clf_id, data_id, fold_id] = v_measure_score(y[test], y_pred)
                if (round_timer == 2):
                    scores[clf_id, data_id, fold_id] = homogeneity_score(y[test], y_pred)
                if (round_timer == 3):
                    scores[clf_id, data_id, fold_id] = completeness_score(y[test], y_pred)

        # Usrednianie wynikow i wyznaczenie jakosci - metryka
        mean_scores = np.mean(scores, axis=2).T
        print("\nMean scores:\n", mean_scores)

        # rangi
        from scipy.stats import rankdata

        ranks = []
        for ms in mean_scores:
            ranks.append(rankdata(ms).tolist())
        ranks = np.array(ranks)
        print("\nRanks:\n", ranks)

        mean_ranks = np.mean(ranks, axis=0)
        print("\nMean ranks:\n", mean_ranks)

        # testy parowe
        from scipy.stats import ranksums

        alfa = .05
        w_statistic = np.zeros((len(clfs), len(clfs)))
        p_value = np.zeros((len(clfs), len(clfs)))

        for i in range(len(clfs)):
            for j in range(len(clfs)):
                w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])



        headers = list(clfs.keys())
        names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
        w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
        w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
        p_value_table = np.concatenate((names_column, p_value), axis=1)
        p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
        print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

        # tabela przewagi

        advantage = np.zeros((len(clfs), len(clfs)))
        advantage[w_statistic > 0] = 1
        advantage_table = tabulate(np.concatenate(
            (names_column, advantage), axis=1), headers)
        print("\nAdvantage:\n", advantage_table)

        # różnice statystycznie znaczące
        significance = np.zeros((len(clfs), len(clfs)))
        significance[p_value <= alfa] = 1
        significance_table = tabulate(np.concatenate(
            (names_column, significance), axis=1), headers)
        print("\nStatistical significance (alpha = 0.05):\n", significance_table)


