import itertools
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from misc import plot_1component, radial_distribution_function


class TrajectoryMatching:

    def __init__(self, outfile_path, basis, basis_parameters, simulation_timestep=1, cutoff=20,
                 every_n_from_output=1, timesteps_in_fit=100, system_style='atomic', op_sys='Linux'):

        self.t = every_n_from_output
        self.n = timesteps_in_fit
        self.outfile_path = outfile_path
        self.output_properties = []
        self.timestep = simulation_timestep
        self.cutoff = cutoff
        self.basis = basis
        self.basis_params = basis_parameters

        self.atom_type_pairs = []
        self.unique_type_pairs = []
        self.box_dimensions = []
        self.box_lo = []
        self.data = []
        self.r = []
        self.v = []
        self.a = []
        self.f = []
        if system_style not in ["atomic", "molecular"]:
            raise ValueError("'system_style' most be either 'atomic' or 'molecular'")
        else:
            self.system_style = system_style
        self.op_sys = op_sys

        self.read_data()

        self.feature_matrix = []
        self.target_vector = []
        self.weights = []

    def read_data(self):
        print("loading data\t")
        t_ = time.time()

        t = -1
        output_timestep = 0
        output_properties = []
        with open(self.outfile_path) as f:
            while t < 1:
                line = f.readline().split()
                if len(line) > 1:
                    if line[1] == "TIMESTEP":
                        t += 1
                        if output_timestep == 0:
                            output_timestep = int(f.readline().split()[0])
                        else:
                            output_timestep = int(f.readline().split()[0]) - output_timestep
                            self.timestep *= output_timestep
                    if line[0] == "ITEM:" and line[1] == "ATOMS":
                        output_properties = np.array(line[2:])
                        self.output_properties = output_properties
        self.timestep *= self.t

        with open(self.outfile_path) as f:
            for l, line in enumerate(f):
                pass
        l += 1

        with open(self.outfile_path) as f:
            t = -1
            pbar = tqdm(total=self.n)
            for i in range(l):
                line = f.readline().split()
                if len(line) == 2:
                    if line[1] == "TIMESTEP":
                        t += 1
                        if len(self.data) == self.n:
                            break
                        if t % self.t == 0:
                            pbar.update(1)
                            self.data.append([])
                elif len(line) == 6 and t % self.t == 0:
                    if line[1] == "BOX":
                        dimensions = []
                        self.box_lo.append([])
                        for j in range(3):
                            line = f.readline().split()
                            dimensions.append(float(line[1]) - float(line[0]))
                            self.box_lo[-1].append(float(line[0]))
                        i += 3
                        self.box_dimensions.append(dimensions)
                elif len(line) == len(output_properties) and t % self.t == 0:
                    self.data[-1].append(np.array(line).astype(float)[1:])
        pbar.close()
        self.box_dimensions = np.array(self.box_dimensions)
        self.data = np.array(self.data)
        self.box_lo = np.array(self.box_lo)

        print(np.round(time.time() - t_, 2), 's')

    def average_force(self, force):
        periods = self.t
        weights = np.ones(periods) / periods
        return np.convolve(force, weights, mode='valid')

    def _reduce_to_centre_of_mass(self, mol_id):
        data = np.array(self.data)
        n = np.array(self.data).shape[0]
        data = data[data[:, :, 1] == mol_id]
        data = np.reshape(data, (n, -1, data.shape[-1]))
        columns = np.nonzero(~ ((self.output_properties[1:] == 'mass') | (self.output_properties[1:] == 'xu') | (
                self.output_properties[1:] == 'yu') | (self.output_properties[1:] == 'zu')))
        data = np.delete(data, columns, axis=2)

        m = np.sum(data[0], axis=0)[0]
        data = np.swapaxes(data, 1, 2)
        data[:, 1:] = data[:, 1:] * data[0, 0] / m
        data = np.swapaxes(data, 1, 2)
        data = np.sum(data, axis=1)

        return data

    def _construct_features(self, packed_t_data):
        np.seterr(all='ignore')

        r_t = packed_t_data[0]
        box_dimensions = packed_t_data[1]
        x_train = []
        y_train = []
        z_train = []

        lx, ly, lz = box_dimensions[0], box_dimensions[1], box_dimensions[2]
        relative_pos_x = d_x = -np.subtract.outer(r_t[:, 0], r_t[:, 0])
        relative_pos_y = d_y = -np.subtract.outer(r_t[:, 1], r_t[:, 1])
        relative_pos_z = d_z = -np.subtract.outer(r_t[:, 2], r_t[:, 2])
        pbc_relative_pos_x = d_pbc_x = np.where(np.abs(d_x) >= 0.5 * lx, d_x - np.sign(d_x) * lx, relative_pos_x)
        pbc_relative_pos_y = d_pbc_y = np.where(np.abs(d_y) >= 0.5 * ly, d_y - np.sign(d_y) * ly, relative_pos_y)
        pbc_relative_pos_z = d_pbc_z = np.where(np.abs(d_z) >= 0.5 * lz, d_z - np.sign(d_z) * lz, relative_pos_z)
        pbc_dist = np.sqrt(d_pbc_x ** 2 + d_pbc_y ** 2 + d_pbc_z ** 2)

        for pair_type in self.unique_type_pairs:
            for p in self.basis_params:
                force_p = self.basis(pbc_dist, p)
                force_p = np.nan_to_num(force_p, posinf=0, neginf=0, nan=0)
                force_p = np.where(pbc_dist > self.cutoff, 0, force_p)
                force_p = np.where(self.atom_type_pairs != pair_type, 0, force_p)

                force_p_x = np.nan_to_num(force_p * pbc_relative_pos_x / pbc_dist, posinf=0, neginf=0, nan=0)
                force_p_y = np.nan_to_num(force_p * pbc_relative_pos_y / pbc_dist, posinf=0, neginf=0, nan=0)
                force_p_z = np.nan_to_num(force_p * pbc_relative_pos_z / pbc_dist, posinf=0, neginf=0, nan=0)

                x_train.append(np.sum(force_p_x, axis=0))
                y_train.append(np.sum(force_p_y, axis=0))
                z_train.append(np.sum(force_p_z, axis=0))

        return [x_train, y_train, z_train]

    def prepare_training_data(self):
        print("preparing input\t")
        t_ = time.time()

        if self.system_style == "atomic":
            f_columns = np.nonzero(~ ((self.output_properties[1:] == 'fx') | (self.output_properties[1:] == 'fy') |
                                      (self.output_properties[1:] == 'fz')))
            self.f = np.array(np.delete(self.data, f_columns, axis=2))

        if self.system_style == "molecular":
            mol_ids = np.unique(self.data[:, :, int(np.where(self.output_properties[1:] == 'mol')[0])])

            reduced_data = []
            for id in mol_ids:
                reduced_data.append(self._reduce_to_centre_of_mass(id))

            self.data = np.swapaxes(np.array(reduced_data), 0, 1)
            ru = self.data[:, :, 1:4]

            shift_to_centre = np.repeat([np.array(self.box_lo)], np.array(self.data).shape[1], axis=0)
            shift_to_centre = np.swapaxes(shift_to_centre, 0, 1)
            shift_to_unit_cell = np.repeat([np.array(self.box_dimensions)], np.array(self.data).shape[1], axis=0)
            shift_to_unit_cell = np.swapaxes(shift_to_unit_cell, 0, 1)

            self.r = ru - shift_to_centre
            self.r = np.floor(self.r / shift_to_unit_cell) * shift_to_unit_cell
            self.r = ru - self.r

        elif self.system_style == "atomic":
            r_columns = np.nonzero(~ ((self.output_properties[1:] == 'x') | (self.output_properties[1:] == 'y') |
                                      (self.output_properties[1:] == 'z')))
            self.r = np.array(np.delete(self.data, r_columns, axis=2))

        d = self.r[1:] - self.r[:-1]

        box_dimensions_ = np.repeat([np.array(self.box_dimensions)], np.array(self.data).shape[1], axis=0)
        box_dimensions_ = np.swapaxes(box_dimensions_, 0, 1)[1:]
        d = np.where(np.abs(d) >= 0.5 * box_dimensions_, d - np.sign(d) * box_dimensions_, d)

        v = d / self.timestep
        self.v = v
        self.a = (v[1:] - v[:-1]) / self.timestep

        atom_types, atom_types_dict, id = [], {}, 0
        for row in self.data[0]:
            mass = np.round(row[0], 2)
            if mass not in list(atom_types_dict.keys()):
                id += 1
                atom_types_dict[mass] = str(id)
                atom_types.append(atom_types_dict[mass])
            else:
                atom_types.append(atom_types_dict[mass])
        atom_type_pairs = np.array(np.meshgrid(atom_types, atom_types))
        atom_type_pairs = np.char.add(atom_type_pairs[0], atom_type_pairs[1]).T
        for i in range(len(atom_type_pairs)):
            for j in range(len(atom_type_pairs[i])):
                if atom_type_pairs[i][j][0] > atom_type_pairs[i][j][1]:
                    atom_type_pairs[i][j] = atom_type_pairs[i][j][1] + atom_type_pairs[i][j][0]
        self.atom_type_pairs = atom_type_pairs
        self.unique_type_pairs = np.unique(np.ravel(np.array(self.atom_type_pairs)))

        vec_packed_t_data = []
        for t in tqdm(range(len(self.r))):
            vec_packed_t_data.append([self.r[t], self.box_dimensions[t]])

        if self.op_sys == "Linux" or self.op_sys == "UNIX" or self.op_sys == "L":
            train_features = Pool().map(self._construct_features, vec_packed_t_data)
        else:
            train_features = []
            for inp in vec_packed_t_data:
                train_features.append(self._construct_features(inp))
        train_features = np.swapaxes(train_features, 0, 1)
        x_train = train_features[0]
        y_train = train_features[1]
        z_train = train_features[2]
        feature_matrix_t = np.concatenate((x_train, y_train, z_train), axis=2)[1:-1]

        target_vector = np.swapaxes(self.a, 1, 2)
        target_vector = np.reshape(target_vector,
                                   (np.size(self.a, axis=0), np.size(self.a, axis=2) * np.size(self.a, axis=1)))

        m = np.ravel(list([self.data[0, :, 0]]) * 3)
        target_vector = target_vector * m / 4.184e-4

        self.feature_matrix = feature_matrix_t
        self.target_vector = target_vector

        print(np.round(time.time() - t_, 2), "s")

    def b_matrix(self, t):
        t = t + 1  # due to how 'v' is calculated
        cutoff = self.cutoff
        r_t = self.r[t]

        box_dimensions = self.box_dimensions[t]
        box_dimensions = box_dimensions.reshape((1, 3))
        box_dimensions = np.repeat(box_dimensions, self.r.shape[1], axis=0)
        box_dimensions = box_dimensions.reshape((self.r.shape[1], 1, 3))
        box_dimensions = np.repeat(box_dimensions, self.r.shape[1], axis=1)

        r_rel = r_t[:, np.newaxis, :] - r_t[np.newaxis, :, :]
        r_rel = np.where(np.abs(r_rel) >= 0.5 * box_dimensions, r_rel - np.sign(r_rel) * box_dimensions, r_rel)
        r_rel = r_rel ** 2
        r_rel = np.sum(r_rel, axis=2)
        r_rel = np.sqrt(r_rel)
        coeff = (1 - r_rel / cutoff) ** 2
        coeff = np.where(np.abs(r_rel) >= cutoff, 0, coeff)
        coeff = np.where(r_rel == 0, 0, coeff)

        a = coeff.shape[0]
        coeff_blocks = np.zeros((a * 3, a * 3))
        for i in [0, 1, 2]:
            coeff_blocks[i * a:(i + 1) * a, i * a:(i + 1) * a] = coeff[:]

        coeff_blocks = coeff_blocks
        coeff_diag = - np.sum(coeff_blocks, axis=0).reshape((1, -1))
        coeff_diag = np.repeat(coeff_diag, coeff_diag.shape[1], axis=0)
        identity = np.eye(coeff_diag.shape[1]).reshape((coeff_diag.shape[1], -1))
        coeff_diag = - coeff_diag * identity

        b_matrix_t = coeff_blocks + coeff_diag

        return b_matrix_t

    def fit_core(self, t, method):
        target_vector_t = self.target_vector[t]
        if method == 'nve' or method == 'NVE':
            norm = np.matmul(self.feature_matrix[t], self.feature_matrix[t].T)
            projection = np.matmul(self.feature_matrix[t], target_vector_t)
        else:
            b_matrix = self.b_matrix(t)
            b_matrix_t_pinv = np.linalg.pinv(b_matrix)
            norm = np.matmul(b_matrix_t_pinv, self.feature_matrix[t].T)
            norm = np.matmul(self.feature_matrix[t], norm)
            projection = np.matmul(b_matrix_t_pinv, target_vector_t)
            projection = np.matmul(self.feature_matrix[t], projection)

        return [norm, projection]

    def fit(self, method=''):
        t_ = time.time()
        print("Fitting\t", end='')

        projections = []
        norms = []
        # For some reason, pooling works slower here, probably due some matmul multiprocess
        '''if self.op_sys == "Linux" or self.op_sys == "UNIX" or self.op_sys == "L":
            t = range(self.feature_matrix.shape[0])
            norm_projection = Pool().map(self.fit_core, t)

            norms = []
            projections = []
            for idx, res in enumerate(norm_projection):
                norms.append(np.array(norm_projection[idx][0]))
                projections.append(np.array(norm_projection[idx][1]))
            else:'''
        for t, feature_matrix_t in enumerate(self.feature_matrix):
            norm_projection = self.fit_core(t, method)
            norms.append(np.array(norm_projection[0]))
            projections.append(np.array(norm_projection[1]))

        projections, norms = np.array(projections), np.array(norms)
        projection = np.sum(projections, axis=0)
        norm = np.sum(norms, axis=0)
        norm_inverse = np.linalg.inv(norm)
        self.weights = np.matmul(norm_inverse, projection)  # unit conversion to 'LAMMPS real'

        print(np.round(time.time() - t_, 2), 's')

    def refit(self, relative_error_filter=10):
        y = []
        y_ = []
        for t, feature_matrix_t in enumerate(self.feature_matrix):
            y.append(self.target_vector[t])
            y_.append(np.matmul(feature_matrix_t.T, self.weights))
        y, y_ = np.array(y), np.array(y_)

        y_diff = np.fabs(y_ - y) / np.fabs(y)

        outliers = np.array(np.nonzero(y_diff > relative_error_filter)).T
        print(f"Removed {len(outliers)}(/{np.prod(y_diff.shape)}) outliers")

        for outlier in outliers:
            t = outlier[0]
            x_ = outlier[1]
            self.feature_matrix[t, :, x_] = 0
            self.target_vector[t][x_] = 0

        self.fit()

    def fit_gamma(self, T=273):
        if len(self.weights) == 0:
            raise RuntimeError("Run fit() first")
        print("Fitting gamma\t", end='')
        t_ = time.time()

        k = 8.314 / 4184  # kcal/mol-K
        gamma_0 = 0
        gamma_2 = 0
        n = self.a.shape[0]
        for t in range(n):
            b_matrix = self.b_matrix(t)
            b_matrix_pinv = np.linalg.pinv(b_matrix)
            v = np.swapaxes(self.v[t], 0, 1)
            v = np.reshape(v, -1)

            g_2 = np.matmul(b_matrix, v)
            gamma_2 += np.matmul(v.T, g_2)
            f_c = np.matmul(self.feature_matrix[t].T, np.array(self.weights))
            f_diff = (self.target_vector[t] - f_c)
            g_0 = np.matmul(b_matrix_pinv, f_diff)
            gamma_0 += np.matmul(f_diff, g_0)

        gamma_0, gamma_2 = -gamma_0 / n, gamma_2 / n
        gamma_1 = 3 * 2 * self.r.shape[1] * k * T / (self.timestep)

        print(np.round(time.time() - t_, 2), 's')
        return np.roots([gamma_2, gamma_1, gamma_0])[1]

    def best_subset(self, k_list, x, center_y=False, print_coeffs=False, plot=False):
        original_weights = np.array(self.weights).copy()
        original_params = np.array(self.basis_params).copy()
        y_target = self.predict(x)
        offset = 0
        if center_y:
            offset = y_target[-1]
            y_target -= offset
        RSS = {}
        weights = {}
        if type(k_list) == tuple:
            X = []
            for param in k_list:
                if param == 0 or param == -1:
                    Warning("Energy fit might be inaccurate if polynomial basis param is 0 or -1")
                X.append(self.basis(x, param))
            X = np.array(X)
            projection = np.matmul(X, y_target)
            norm = np.matmul(X, X.T)
            norm_inverse = np.linalg.inv(norm)
            w = np.matmul(norm_inverse, projection)
            weights[tuple(k_list)] = w
            self.weights = w
            self.basis_params = k_list
            RSS[tuple(k_list)] = np.sqrt(np.sum((self.predict(x) - y_target) ** 2)) / float(len(x))
        elif type(k_list) == int:
            k_list = [k_list]

        if len(weights) == 0:
            for k in k_list:
                original_params_ = []
                for p_ in original_params:
                    if p_ != 0 and p_ != -1 and p_ >= -15:
                        original_params_.append(p_)
                if k > len(original_params_):
                    k = len(original_params_)
                for params in itertools.combinations(original_params_, k):
                    X = []
                    for param in params:
                        X.append(self.basis(x, param))
                    X = np.array(X)
                    projection = np.matmul(X, y_target)
                    norm = np.matmul(X, X.T)
                    norm_inverse = np.linalg.inv(norm)
                    w = np.matmul(norm_inverse, projection)
                    weights[tuple(params)] = w
                    self.weights = w
                    self.basis_params = params
                    RSS[tuple(params)] = np.sqrt(np.sum((self.predict(x) - y_target) ** 2)) / float(len(x))

        RSS = dict(sorted(RSS.items(), key=lambda item: item[1]))
        key = list(RSS.keys())[0]

        self.weights = weights[key]
        self.basis_params = np.array(key)
        if plot:
            y_fit_ = self.predict(x)
            if len(y_target) == len(x):
                plot_1component(x, y_target, y_fit_, labels=["reduced parameter set", "TM fit"])
            else:
                plot_1component(x, y_fit_, labels=["TM fit"])

        if print_coeffs:
            print("pair_coeff\t1 1 ", end='')
            for p in range(0, -15, -1):
                if p in self.basis_params:
                    if p == 0:
                        print(self.weights[self.basis_params == p][0] - offset, end='  ')
                    else:
                        print(self.weights[self.basis_params == p][0], end='  ')
                elif p == 0 and center_y:
                    print(offset, end='  ')
                else:
                    print('0 ', end='  ')
            print('\n')

        self.basis_params = original_params
        self.weights = original_weights

        return np.array(key), weights[key]

    def predict(self, x):
        number_of_type_pairs = len(self.unique_type_pairs)
        p = len(self.weights)
        number_of_forces = int(p / number_of_type_pairs)
        Y = []
        for i in range(number_of_type_pairs):
            if type(x) == float:
                y = 0
            else:
                y = np.zeros(len(x))
            for j in range(len(self.basis_params)):
                y = y + self.weights[i * number_of_forces + j] * self.basis(x, self.basis_params[j])
            Y.append(y)
        if number_of_type_pairs == 1:
            return np.array(Y[0])
        else:
            return Y

    def predict_energy(self, x_fit, x_plot, best_subset_=0):
        if best_subset_ == 0:
            best_subset_ = max(2, len(self.basis_params) - 3)
        pars, weights_ = self.best_subset(best_subset_, x_fit)

        weights = np.zeros(len(self.basis_params))
        for idx, param in enumerate(pars):
            weights[self.basis_params == param] += weights_[idx]

        number_of_type_pairs = len(self.unique_type_pairs)
        p = len(weights)
        number_of_forces = int(p / number_of_type_pairs)
        Y = []
        for i in range(number_of_type_pairs):
            if type(x_plot) == float:
                y = 0
            else:
                y = np.zeros(len(x_plot))
            for j, param in enumerate(self.basis_params):
                w = weights[i * number_of_forces + j]
                if w == 0:
                    continue
                if param == 0:
                    # continue
                    y = y + w * x_plot
                elif param == -1:
                    # y = y + w * np.log(x)
                    continue
                else:
                    y = y + w / np.abs(param + 1) * x_plot ** (param + 1)
            Y.append(y)
        if number_of_type_pairs == 1:
            return np.array(Y[0])
        else:
            return Y

    def write_pair_table(self, energy_fit_x, n, outfile_path='pair.table', energy_fit_params=0):
        x = np.arange(1, n + 1, 1)
        x = np.sqrt(x)
        x *= self.cutoff / x[-1]
        f = self.predict(x)
        if energy_fit_params != 0:
            e = self.predict_energy(energy_fit_x, x)
        else:
            e = self.predict_energy(energy_fit_x, x, energy_fit_params)

        file_ = open(outfile_path, 'w')
        file_.write("1-1\n")
        file_.write(f"N {len(x)}\tRSQ {x[0]} {x[-1]}\n\n")

        for idx, x_ in enumerate(x):
            file_.write(f"{idx + 1} {x_} {e[idx]} {f[idx]}\n")
        file_.close()

    def plot_rdf(self, r_max=17.5, dr=0.1, plot=True, outfile_path=''):
        g_list = []
        radii = []
        for t in range(len(self.r)):
            x = self.r[t, :, 0]
            y = self.r[t, :, 1]
            z = self.r[t, :, 2]
            s = np.mean(self.box_dimensions[t])
            # x, y, z = augment(x, y, z, s)
            g, radii, indices = radial_distribution_function(x, y, z, s * 3, r_max, dr)
            g_list.append(g)
        g_list = np.array(g_list)
        g_ave = np.mean(g_list, axis=0)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(radii, g_ave)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            if outfile_path != '':
                plt.savefig(outfile_path, bbox_inches='tight')
        return radii, g_ave
