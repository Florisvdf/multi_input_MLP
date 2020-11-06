import multiprocessing
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import scale, minmax_scale
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import QuantileTransformer


class CID:

    def __init__(self, data, graph=None, n_bins='sqrt', scale_data=True, discretize=False, kwargs=None,
                 data_std_threshold=None):

        """
        Base class for Covered Information Disentanglement.
        :param data: Data
        :param graph: if you have a pre-specified graph, pass here.
        :param n_bins: Number of bins to discretize data into
        :param scale_data: Whether to scale the data using z-tranformation
        :param discretize: whether or not to discretize. The CID nice intuitive
        properties only work for discrete data, so we recommend always
        discretizing your data
        :param kwargs: Pass here the arguments for the network structure learning
        :param data_std_threshold: if None, all data is used to compute CID,
        else, all data xi>data_std_threshold * std will be excluded. This can
        help getting better network inference because discretization will not
        be affected by outliers
        """

        df = pd.DataFrame(data)

        self.n_feats = data.shape[1]

        if scale_data:
            df = pd.DataFrame(scale(df.values, with_std=True), columns=df.columns)

        self.data = df

        self.n_bins = self.convert_bins(n_bins)

        self.cont_feats, self.disc_feats = self.identify_continuous_features()

        if data_std_threshold is not None:
            self.remove_extreme_values(data_std_threshold)

        if discretize:
            self.deltas = None

            self.discretize_data(self.n_bins)
            if scale_data:
                self.data.iloc[:, self.cont_feats] = scale(
                    self.data.iloc[:, self.cont_feats].values, with_std=True)
            #
        if len(self.disc_feats) > 0:
            self.data.iloc[:, self.disc_feats] = scale(
                self.data.iloc[:, self.disc_feats].values, with_std=True)

        # means = np.mean(self.data.values, axis=0)
        # diffs = [self.values_table[i] - means[i] for i in range(n_feats)]
        # smallest_diffs = [diffs[i][np.argmin(np.abs(diffs[i]))] for i in range(n_feats)]
        # self.data = pd.DataFrame(self.data.values - (smallest_diffs * np.sign(smallest_diffs)),
        #                          columns=self.data.columns, index=self.data.index)

        self.values_table = self.get_unique_values_table()

        self.deltas = [self.values_table[i][1]-self.values_table[i][0]
                       for i in range(self.n_feats)]

        if graph is None:
            self.create_graph(kwargs)
        else:
            self.graph = graph

        self.neighbors = self.get_neighbors_list()

        self.entropies = np.zeros((data.shape[1],))

    def identify_continuous_features(self):

        cont_feats = []
        floats = [isinstance(self.data.dtypes[i], np.float64) for i in range(self.n_feats)]
        for i in range(self.n_feats):
            if floats[i]:
                cont_feats.append(i)
            elif len(np.unique(self.data.iloc[:, i])) / self.data.shape[0] > 0.05:
                cont_feats.append(i)

        discrete_feats = np.arange(self.n_feats)

        discrete_feats = np.delete(discrete_feats, cont_feats)

        return cont_feats, discrete_feats

    def get_unique_values_table(self):

        values_table = []
        for i in range(self.n_feats):
            if i in self.cont_feats:
                values_table.append(np.linspace(
                    np.min(self.data.iloc[:, i]),
                    np.max(self.data.iloc[:, i]),
                    self.n_bins))
                # values_table.append(np.linspace(
                #     np.min(self.data.iloc[:, i]),
                #     np.max(self.data.iloc[:, i]),
                #     self.n_bins * 2)[1::2])

            else:
                values_table.append(np.unique(self.data.iloc[:, i]))
        return values_table

    def compute_values_table(self, n_bins=None):

        pass

    def get_neighbors_list(self):

        neighbors = []

        for i, row in enumerate(self.graph):

            non_zeros_inds = np.argwhere(row != 0).flatten()

            i_ind = np.argwhere(non_zeros_inds == i).flatten()[0]

            if len(non_zeros_inds) > 1:
                non_zeros_inds[[0, i_ind]] = non_zeros_inds[[i_ind, 0]]  # for easier access later on

            neighbors.append(np.array(non_zeros_inds))

        return neighbors

    def create_graph(self, **kwargs):

        pass

    def convert_bins(self, n_bins):

        if isinstance(n_bins, str):

            if 'sqrt' in n_bins:
                n_bins = int(np.sqrt(self.data.shape[0]))

        elif isinstance(n_bins, int):

            n_bins = n_bins

        return n_bins

    def discretize_data(self, n_bins, bound_slack=0.0):

        """
        Data discretization
        :param n_bins: number of bins to fit the data into
        :param bound_slack: the amount of slack to give to the right-most
        and left-most bin edges
        :return:
        """

        # self.values_table = [np.linspace(
        #     np.min(col) - np.abs(np.min(col)) * bound_slack,
        #     np.max(col) + np.abs(np.max(col)) * bound_slack,
        #     n_bins * 2)[1::2] for col in self.data.values.T]

        # self.deltas = np.array(
        #     [self.values_table[i][1] - self.values_table[i][0] for i in range(self.data.shape[1])])

        self.data.iloc[:, self.cont_feats] = self.data.iloc[:, self.cont_feats].apply(
            lambda col: pd.cut(col, bins=n_bins,
                               labels=np.linspace(
                                   np.min(col) - np.abs(np.min(col)) * bound_slack,
                                   np.max(col) + np.abs(np.max(col)) * bound_slack,
                                   n_bins * 2)[1::2]))

    def get_neighbors(self, ind):

        neighbors = self.graph[ind, :].nonzero()

        return neighbors

    def remove_extreme_values(self, std_threshold=2):

        stds = np.std(self.data.values, axis=0)
        rows_to_remove = np.hstack([np.argwhere(np.abs(self.data.values[:, i]) >
                                                std_threshold * stds[i]).flatten()
                                    for i in range(self.data.shape[1])])
        rows_to_remove = np.unique(rows_to_remove)
        self.data = self.data.drop(self.data.index[rows_to_remove])

    def fit(self, X=None, y=None, n_samples=0.3):

        pass

    @staticmethod
    def convert_n_samples(n_samples, n_instances):

        if isinstance(n_samples, int):

            n_samples = n_samples

        elif isinstance(n_samples, float):

            n_samples = int(n_samples * n_instances)

        return n_samples

    @staticmethod
    def select_data_by_id(data, ids):

        if isinstance(data, pd.DataFrame):
            if isinstance(ids, pd.Index):
                data = data.loc[ids, :].values
            else:
                data = data.iloc[ids, :].values
        else:
            data = data[ids, :]

        return data

    @staticmethod
    def select_data_cols(data, cols):

        if isinstance(data, pd.DataFrame):
            if isinstance(cols[0], int):
                data = data.iloc[:, cols].values
            else:
                data = data.loc[:, cols].values
        else:
            data = data[:, cols]

        return data

    @staticmethod
    def safe_log(x):
        if x > 0:
            log = np.log(x)
        else:
            log = 0

        return log

    @staticmethod
    def safe_divide(a, b):
        return np.array([a[i] / b[i] if np.logical_and(a[i] != 0, b[i] != 0) else -1
                         for i in range(len(a))])


class Normal:

    def __init__(self):

        pass

    @staticmethod
    def compute_prob_uni_gaussian(x, mean, cov):

        """
        Computes univariate gaussian probability
        :param x: value of x
        :param mean: mean of the distribution
        :param cov: covariance of the distribution = std**2
        :return: prob(x), X~N(mean, cov)
        """

        scalar = 1 / (np.sqrt(2 * np.pi * cov))
        exp_term = ((x - mean) ** 2 / cov)
        return scalar * np.exp(-0.5 * exp_term)

    @staticmethod
    def prob_gaussian(x, mean, sqrt_cov_det, precision):

        p = len(x)

        denom = 1 / ((2 * np.pi) ** (p / 2) * sqrt_cov_det)

        if len(x.shape) == 1:

            exp_arg = reduce(np.matmul, [(x - mean).T, precision, (x - mean)])

        else:

            exp_arg = np.array([reduce(np.matmul, [(x[:, i] - mean).T, precision, (x[:, i] - mean)])
                                for i in range(x.shape[1])])
        return denom * np.exp(-0.5 * exp_arg)

    @staticmethod
    def univariate_gaussian_exponent(x, mean, standard_deviation):

        mean_term = (x - mean) ** 2

        return - mean_term / standard_deviation ** 2

    @staticmethod
    def multivariate_gaussian_exponent(x, mean, precision):

        right = (x - mean)
        left = right.T

        n_samples = x.shape[0]

        if n_samples == 1:

            exponent = -0.5 * reduce(np.matmul, [left, precision, right])
        else:
            exponent = np.array([-0.5 * reduce(np.matmul, [left[:, i].T, precision, right[i, :]])
                                 for i in range(n_samples)])

        return exponent


class CIDGmm(CID, Normal):

    def __init__(self, data, y=None, n_bins='sqrt', scale_data=False, discretize=False, redund_correction=False,
                 threshold_precision=0.08, random_state=0, data_std_threshold=None, empirical_mi=False, kwargs=None):

        """
        Initializes CID using Gaussian Markov Model.
        :param data: Dataset
        :param n_bins: number of bins to discretize data.
        Pass int or array if different number of bins is desired
        :param threshold_precision: value below which to consider entries in the
        estimated precision as 0. Precision entries that are 0 encode independence
        between the features.
        :param redund_correction: whether to apply redundant information correction in the mutual info (see
        R. Ince: The Partial Entropy Decomposition: Decomposing multivariate entropy and mutual
        information via pointwise common surprisal 2017 arxiv:1702.01591)
        :param empirical_mi: whether to use non-parametric measure of mutual information with output (False if not)
        :param data_std_threshold: All values np.abs(x_i)>std * data_std_threshold will be excluded.
        This can help getting a more accurate approximation of the density function by
        removing outliers.
        :param kwargs: arguments to pass to graphical lasso. See sklearn.covariance for more info
        """

        self.threshold_precision = threshold_precision

        self.precision = None
        self.mean = None
        self.nu = None
        self.empirical_mi = empirical_mi
        self.redund_correction = redund_correction

        self.random_state = random_state

        self.y = y
        data_ = data.copy()

        if y is not None:
            if isinstance(data, pd.DataFrame):
                data_['y'] = y
            else:
                data_ = np.hstack([data_, y])

        CID.__init__(self, data=data_, n_bins=n_bins, scale_data=scale_data,
                     discretize=discretize,
                     data_std_threshold=data_std_threshold, kwargs=kwargs)

        self.y_pot_tensor = self.compute_joint_y_pot_tensor()

    def compute_discretized_marginals(self):

        marginals = np.zeros((self.data.shape[1],))

        for i in range(len(marginals)):
            delta = self.deltas[i]
            mean = self.mean[i]
            var = self.covariance[i, i]
            values = self.values_table[i]
            marginals[i] += np.sum(self.compute_prob_uni_gaussian(values, mean, var))
            marginals[i] *= delta

        return marginals

    def compute_discretized_marginals_from_two_dim_distribution(self):

        double_marginals = np.zeros((self.data.shape[1], self.data.shape[1]))

        for i in range(len(double_marginals)):

            for j in range(i + 1, len(double_marginals)):
                delta = np.prod([self.deltas[i], self.deltas[j]])
                mean = self.mean[[i, j]]
                cov = self.covariance[np.ix_([i, j], [i, j])]
                cov_det = np.linalg.det(cov) ** 0.5
                precision = np.linalg.pinv(self.covariance[np.ix_([i, j], [i, j])])
                values = np.array(np.meshgrid(self.values_table[i], self.values_table[j])).reshape(2, -1)

                double_marginals[i, j] = np.sum(self.prob_gaussian(values, mean, cov_det, precision)) * delta

        return double_marginals

    def plot_joint_discretized_density(self, i, j):

        import matplotlib.pyplot as plt
        fig = plt.figure()
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        ax = fig.add_subplot(111, projection='3d')
        delta = np.prod([self.deltas[i], self.deltas[j]])
        mean = self.mean[[i, j]]
        cov = self.covariance[np.ix_([i, j], [i, j])]
        cov_det = np.linalg.det(cov) ** 0.5
        precision = self.precision[np.ix_([i, j], [i, j])]
        values = np.array(np.meshgrid(self.values_table[i], self.values_table[j])).reshape(2, -1)
        values_z = self.prob_gaussian(values, mean, cov_det, precision) * delta
        xx = values[0, :]
        yy = values[1, :]
        offset = values_z + np.abs(np.min(values_z))
        fracs = offset.astype(float) / offset.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        color_values = cm.jet(norm(fracs.tolist()))

        for row in values_z:
            print(row)
        print(np.sum(values_z))
        ax.bar3d(xx.flatten(),
                 yy.flatten(), np.zeros(len(values_z)), 1, 1, values_z, alpha=1,
                 color=color_values)
        plt.show()

    def create_graph(self, kwargs):

        self.compute_gaussian_cov(**kwargs)

        self.graph = np.array(np.abs(self.precision) > 0.08).astype(int)

    def compute_n_neighbors(self):

        n_neighbors = [len(nns) for nns in self.neighbors]

        return n_neighbors

    def compute_gaussian_cov(self, **kwargs):

        """
        Graphical LASSO for estimating the network graph, covariance and precision
        :param kwargs: Check sklearn.GraphicalLassoCV for the parameters
        :return:
        """
        n_proc = multiprocessing.cpu_count()
        graph_lasso = GraphicalLassoCV(**kwargs)

        graph_lasso.fit(self.data)

        estimated_mean = graph_lasso.location_
        estimated_precision = graph_lasso.precision_
        data_prec = np.linalg.pinv(np.cov(self.data.T))
        self.mean = estimated_mean
        self.precision = estimated_precision
        self.precision[np.abs(self.precision) < 0.08] = 0
        self.nu = np.matmul(self.precision, self.mean)
        self.covariance = np.linalg.pinv(self.precision)

    def mutual_info_with_y(self, sample_ids):

        """
        Computes the empirical mutual info for each variable with the output.
        :param sample_ids: The ids of the sampled rows
        :return: MI(x_i, y) for all i in [1, N_features]
        """

        data = self.select_data_by_id(self.data, sample_ids)

        p_xs = np.array([self.compute_prob_uni_gaussian(data[:, i], self.mean[i], self.covariance[i, i]) for i in
                         range(data.shape[1])])
        p_x_y = np.array([self.prob_gaussian(data[:, [i, -1]].T, self.mean[[i, -1]],
                                             np.linalg.det(self.covariance[np.ix_([i, -1], [i, -1])]) ** 0.5,
                                             np.linalg.pinv(self.covariance[np.ix_([i, -1], [i, -1])])) for i in
                          range(data.shape[1] - 1)])
        h = np.log(p_x_y / (p_xs[:-1] * p_xs[-1]))
        if self.redund_correction:
            h = h.clip(min=0)

        h = np.sum(h, axis=1) / data.shape[0]
        # h = np.sum(np.log(p_x_y / (p_xs[:-1] * p_xs[-1])), axis=1) / data.shape[0]

        return h

    def sample_entropy_expectation(self, samples_ids):

        """
        Computes the empirical expected term in the CID value
        :param samples_ids: ids of the sampled rows
        :return:
        """

        data = self.select_data_by_id(self.data, samples_ids)

        n_instances = len(samples_ids)

        n_feats = self.data.shape[1]

        sampled_entropies = np.zeros((n_feats - 1,))

        for sample_row in data:

            for feat in range(n_feats - 1):
                if self.redund_correction:
                    sampled_entropies[feat] += np.max([-self.mutual_infos_y[feat],
                                                       self.compute_sample_covered_term_new(feat, sample_row)])
                else:
                    sampled_entropies[feat] += self.compute_sample_covered_term_new(feat, sample_row)

        sampled_entropies /= n_instances

        return sampled_entropies

    def compute_joint_y_pot_tensor(self):

        """
        Computes a potential tensor for each feature between each value of the feature
        and each value of y, yielding a matrix (n_feats x n_bins x n_bins) where the
        feature values vary over rows and the y values vary over columns
        :return: y_pot_tensor
        """

        n_bins = self.n_bins

        # y_pot_tensor = np.zeros((self.data.shape[1] - 1, n_bins, n_bins))

        y_pot_tensor = []

        for feat in range(self.data.shape[1] - 1):
            feat_y_pair_values = np.array(np.meshgrid(self.values_table[-1], self.values_table[feat])).reshape(2, -1)

            exponent = -0.5 * np.prod(feat_y_pair_values, axis=0) * self.precision[feat, -1]

            y_pot_tensor.append(np.exp(exponent.reshape(len(self.values_table[feat]),
                                                               len(self.values_table[-1]))))

        return y_pot_tensor

    def compute_xi_exclude_xj_pot_array(self, i_pos, j_pos, x_sample):

        """
        Computes the potential of feature x_i while excluding the
        potential terms that involve feature x_j
        :param i_pos: feature index whose potentials are to be computed
        :param j_pos: feature index whose potentials are not to be included
        :param x_sample: sampled data row
        :return:
        """

        nn = self.neighbors[i_pos]
        if j_pos == -1:
            j_pos = len(x_sample) - 1
        if i_pos == -1:
            i_pos = len(x_sample) - 1
        nn = nn[np.logical_and(nn != j_pos, nn != i_pos)]

        x_i_values = self.values_table[i_pos]

        x_i_only_pot = self.compute_single_var_pot_exponent(i_pos, x_i_values)

        cross_pot_array = -0.5 * np.sum(np.outer(self.precision[i_pos, nn] * x_sample[nn], x_i_values), axis=0)

        return np.exp(x_i_only_pot + cross_pot_array)

    def compute_single_var_pot_exponent(self, i_pos, x_i_values):

        """
        Computes the potential exponent that involves only feature x_i
        :param i_pos: index of the feature
        :param x_i_values: values of the feature x_i for the sampled rows
        :return:
        """

        x_i_only_pot = -0.5 * (x_i_values ** 2 * self.precision[i_pos, i_pos] -
                               2 * x_i_values * self.nu[i_pos])

        return x_i_only_pot

    def compute_sample_covered_term_new(self, ind, x_sample):

        """
        Computes expected value in covered info term.
        D is the array of potentials between feature x_i and its neighbors excluding y
        E is the array of potentials between y and its neighbors excluding x_i
        F is the matrix of potentials between y and x_i for all values y, x_i
        :param ind: index of feature x_i
        :param sampled_x: values for each feature sampled from data
        :return: Covered info numerator term for this sample
        """

        y_value_pos = int((x_sample[-1] - self.values_table[-1][0]) / self.deltas[-1] + 0.001)
        x_value_pos = int((x_sample[ind] - self.values_table[ind][0]) / self.deltas[ind] + 0.001)

        F = self.y_pot_tensor[ind]

        first_term = np.log(F[x_value_pos, y_value_pos])

        D = self.compute_xi_exclude_xj_pot_array(i_pos=ind, j_pos=-1,
                                                 x_sample=x_sample)

        E = self.compute_xi_exclude_xj_pot_array(i_pos=-1, j_pos=ind,
                                                 x_sample=x_sample)

        numerator = reduce(np.matmul, [D.T, F, E])

        f_Y = F[:, y_value_pos]
        f_X_i = F[x_value_pos, :].T

        demon_1 = np.matmul(D[np.newaxis, :], f_Y)[0]
        demon_2 = np.matmul(E[np.newaxis, :], f_X_i)[0]

        final_term = first_term + np.log(numerator / (demon_1 * demon_2))

        return final_term

    def fit(self, X=None, y=None, n_samples=0.5):

        """
        Computes the CID value for each feature
        :param X: data
        :param y: target values
        :param n_samples: Number of samples to use when computing the empirical
        expectation term in the CID values
        :return: covered information for each feature
        """

        if X is None:
            X = self.data
        else:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

        ids = np.arange(X.shape[0])

        np.random.seed(self.random_state)

        n_samples = int(n_samples * X.shape[0])

        sample_ids = np.random.choice(ids, n_samples)

        entros = [mutual_info_score(self.data.iloc[:, i], self.data.iloc[:, -1])
                  for i in range(self.data.shape[1] - 1)]

        mutual_infos_y = np.around(self.mutual_info_with_y(sample_ids=sample_ids), 4)

        mutual_infos_y[mutual_infos_y<0.005] = 0

        self.mutual_infos_y = mutual_infos_y

        sampled_entropy_expectation = - self.sample_entropy_expectation(samples_ids=sample_ids)

        sampled_entropy_expectation = np.around(sampled_entropy_expectation, 6)

        if self.empirical_mi:
            denom = entros
        else:
            denom = mutual_infos_y

        covered_term = self.safe_divide(sampled_entropy_expectation, denom)

        covered_information = 1 + covered_term

        covi = pd.DataFrame(covered_information, index=self.data.columns[:-1])
        mi = pd.DataFrame(mutual_infos_y, index=self.data.columns[:-1])
        prec = pd.DataFrame(self.precision, index=self.data.columns, columns=self.data.columns)
        cov = pd.DataFrame(self.covariance, index=self.data.columns, columns=self.data.columns)

        covered_information[covered_information < 0] = 0
        covered_information[covered_information > 1] = 1
        covered_information = covered_information.clip(min=0, max=.95)

        return covered_information, mutual_infos_y