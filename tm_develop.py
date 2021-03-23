import numpy as np
import time
import itertools
from sklearn import linear_model
from multiprocessing import Pool
from misc import plot_1component


class TrajectoryMatching:

    def __init__(self, outfile_path, simulation_timestep, cutoff, basis, basis_parameters,
                 every_n_from_output=1, timesteps_in_fit=100, system_style='atomic', op_sys='Linux'):

        self.t = every_n_from_output
        self.n = timesteps_in_fit
        self.outfile_path = outfile_path
        self.output_properties = []
        self.timestep = simulation_timestep * every_n_from_output
        self.cutoff = cutoff
        self.basis = basis
        self.basis_params = basis_parameters

        self.atom_type_pairs = []
        self.unique_type_pairs = []
        self.box_dimensions = []
        self.box_lo = []
        self.data = []
        self.r = []
        self.ru = []
        self.a = []
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
        #print("Reading data\t", end="")
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

        data = []
        with open(self.outfile_path) as f:
            l = len(f.readlines())

        with open(self.outfile_path) as f:
            t = -1
            for i in range(l):
                line = f.readline().split()
                if len(line) == 2:
                    if line[1] == "TIMESTEP":
                        t += 1
                        if len(data) == self.n:
                            break
                        if t % self.t == 0:
                            data.append([])
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
                    data[-1].append(np.array(line).astype(float)[1:])

        self.data = np.array(data)
        if self.system_style == "molecular":
            mol_ids = np.unique(self.data[:, :, int(np.where(output_properties[1:] == 'mol')[0])])
            if self.op_sys == "Linux" or self.op_sys == "UNIX" or self.op_sys == "L":
                reduced_data = Pool().map(self._reduce_to_centre_of_mass, mol_ids)
            else:
                reduced_data = []
                for id in mol_ids:
                    reduced_data.append(self._reduce_to_centre_of_mass(id))
            self.data = np.swapaxes(np.array(reduced_data), 0, 1)
            self.ru = self.data[:, :, 1:4]

            shift_to_centre = np.repeat([np.array(self.box_lo)], np.array(self.data).shape[1], axis=0)
            shift_to_centre = np.swapaxes(shift_to_centre, 0, 1)
            shift_to_unit_cell = np.repeat([np.array(self.box_dimensions)], np.array(self.data).shape[1], axis=0)
            shift_to_unit_cell = np.swapaxes(shift_to_unit_cell, 0, 1)

            self.r = self.ru - shift_to_centre
            self.r = np.floor(self.r / shift_to_unit_cell) * shift_to_unit_cell
            self.r = self.ru - self.r

        elif self.system_style == "atomic":
            r_columns = np.nonzero(~ ((self.output_properties[1:] == 'x') | (self.output_properties[1:] == 'y') |
                                      (self.output_properties[1:] == 'z')))
            self.r = np.array(np.delete(data, r_columns, axis=2))
            ru_columns = np.nonzero(~ ((self.output_properties[1:] == 'xu') | (self.output_properties[1:] == 'yu') |
                                       (self.output_properties[1:] == 'zu')))
            self.ru = np.array(np.delete(data, ru_columns, axis=2))

        d = self.r[1:] - self.r[:-1]
        box_dimensions = np.repeat([np.array(self.box_dimensions)], np.array(self.data).shape[1], axis=0)
        box_dimensions = np.swapaxes(box_dimensions, 0, 1)[1:]
        d = np.where(np.abs(d) >= 0.5 * box_dimensions, d - np.sign(d) * box_dimensions, d)

        v = d / self.timestep
        self.a = (v[1:] - v[:-1]) / self.timestep

        #print(np.round(time.time() - t_, 2), "s")

    def _reduce_to_centre_of_mass(self, mol_id):
        data = self.data
        data = data[data[:, :, 1] == mol_id]
        data = np.reshape(data, (np.array(self.data).shape[0], -1, data.shape[-1]))
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

    def prepare_training_data(self, method="bayesian"):
        #print("Preparing input\t", end="")
        t_ = time.time()

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
        for t in range(len(self.r)):
            vec_packed_t_data.append([self.r[t], self.box_dimensions[t]])

        if self.op_sys == "Linux" or self.op_sys == "UNIX" or self.op_sys == "L":
            train_features = Pool().map(self._construct_features, vec_packed_t_data)
        else:
            train_features = []
            for input in vec_packed_t_data:
                train_features.append(self._construct_features(input))
        train_features = np.swapaxes(train_features, 0, 1)
        x_train = train_features[0]
        y_train = train_features[1]
        z_train = train_features[2]
        feature_matrix_t = np.concatenate((x_train, y_train, z_train), axis=2)[1:-1]

        target_vector = np.swapaxes(self.a, 1, 2)
        target_vector = np.reshape(target_vector,
                                   (np.size(self.a, axis=0), np.size(self.a, axis=2) * np.size(self.a, axis=1)))

        m = np.ravel(list([self.data[0, :, 0]]) * 3)
        target_vector = target_vector * m

        if method == 'simple':
            feature_matrix_t = np.swapaxes(feature_matrix_t, 0, 1)
            feature_matrix_t = np.reshape(feature_matrix_t, (np.size(feature_matrix_t, axis=0),
                                                             np.size(feature_matrix_t, axis=1) * np.size(
                                                                 feature_matrix_t, axis=2)))
            target_vector = np.ravel(self.target_vector)

        if method == "bayesian":
            feature_matrix_t = np.swapaxes(feature_matrix_t, 0, 1)
            feature_matrix_t = np.sum(feature_matrix_t, axis=1)
            target_vector = np.sum(target_vector, axis=0)

        self.feature_matrix = feature_matrix_t.T
        self.target_vector = target_vector

        #print(np.round(time.time() - t_, 2), "s\n")

    def fit(self, method='ElasticNet', alpha=1e-6, l1_ratio=0.75):

        if method == "simple":
            projection = np.matmul(np.array(self.feature_matrix).T, self.target_vector)
            norm = np.matmul(np.array(self.feature_matrix).T, np.array(self.feature_matrix))
            norm_inverse = np.linalg.inv(norm)
            self.weights = np.matmul(norm_inverse, projection) / 4.184e-4  # ` unit conversion from kcal

        elif method == "Ridge":
            regressor = linear_model.Ridge(alpha=alpha, fit_intercept=True, normalize=True, max_iter=10000)
            regressor.fit(self.feature_matrix, self.target_vector)
            self.weights = regressor.coef_ / 4.184e-4
        elif method == "Lasso":
            regressor = linear_model.Lasso(alpha=alpha, fit_intercept=True, normalize=True, max_iter=10000)
            regressor.fit(self.feature_matrix, self.target_vector)
            self.weights = regressor.coef_ / 4.184e-4
        else:
            regressor = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, normalize=True, max_iter=10000)
            regressor.fit(self.feature_matrix, self.target_vector)
            self.weights = regressor.coef_ / 4.184e-4

    def best_subset(self, k_list, x, center_y=False):
        original_weights = np.array(self.weights).copy()
        original_params = np.array(self.basis_params).copy()
        y_target = self.predict(x)
        if center_y:
            y_target -= y_target[-1]
        RSS = {}
        weights = {}
        for k in k_list:
            for params in itertools.combinations(original_params, k):
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
                RSS[tuple(params)] = np.sqrt(np.sum((self.predict(x) - y_target / 4.184e-4) ** 2)) / float(len(x))

        RSS = dict(sorted(RSS.items(), key=lambda item: item[1]))
        key = list(RSS.keys())[0]

        self.weights = weights[key]
        self.basis_params = np.array(key)
        y_fit_ = self.predict(x)
        if len(y_target) == len(x):
            plot_1component(x, y_fit_, y_target, labels=["TM fit", "reduced parameter set"])
        else:
            plot_1component(x, y_fit_, labels=["TM fit"])

        #print(f"Non-zero weight powers: {self.basis_params}")
        print("pair_coeff\t1 1 ", end='')
        for p in range(0, min(original_params) - 1, -1):
            if p in self.basis_params:
                print(self.weights[self.basis_params == p][0], end='  ')
            else:
                print('0 ', end='  ')
        print('\n')

        self.basis_params = original_params
        self.weights = original_weights

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
