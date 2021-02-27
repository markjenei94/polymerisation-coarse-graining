import numpy as np
import time
from multiprocessing import Pool


class TrajectoryMatching:

    def __init__(self, outfile_path, simulation_timestep, cutoff, basis, basis_parameters,
                 every_n_from_output=1, timesteps_in_fit=100, system_style='atomic'):

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
        self.read_data()

        self.feature_matrix_t = []
        self.target_vector = []
        self.weights = []

    def read_data(self):
        print("Reading data\t", end="")
        t_ = time.time()

        # Gather information
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

        # TODO: rewrite self.atom_type_pairs stuff

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
            reduced_data = Pool().map(self._reduce_to_centre_of_mass, mol_ids)
            self.data = np.swapaxes(np.array(reduced_data), 0, 1)
            self.ru = self.data[:, :, 1:4]

            shift_to_centre = np.repeat([np.array(self.box_lo)], np.array(self.data).shape[1], axis=0)
            shift_to_centre = np.swapaxes(shift_to_centre, 0, 1)
            shift_to_unit_cell = np.repeat([np.array(self.box_dimensions)], np.array(self.data).shape[1], axis=0)
            shift_to_unit_cell = np.swapaxes(shift_to_unit_cell, 0, 1)

            self.r = self.ru - shift_to_centre
            self.r = (self.r / shift_to_unit_cell).astype(int) * shift_to_unit_cell
            self.r = self.ru - self.r + shift_to_centre

        elif self.system_style == "atomic":
            r_columns = ~ ((self.output_properties[1:] == 'x') | (self.output_properties[1:] == 'y') |
                           (self.output_properties[1:] == 'z'))
            self.r = np.array(np.delete(data, r_columns, axis=2))
            ru_columns = ~ ((self.output_properties[1:] == 'xu') | (self.output_properties[1:] == 'yu') |
                            (self.output_properties[1:] == 'zu'))
            self.ru = np.array(np.delete(data, ru_columns, axis=2))

        v = (self.ru[1:] - self.ru[:-1]) / self.timestep
        self.a = (v[1:] - v[:-1]) / self.timestep

        print(np.round(time.time() - t_, 2), "s")

    def _reduce_to_centre_of_mass(self, mol_id):
        data = self.data
        data = data[np.where(data[:, :, 1] == mol_id)]
        data = np.reshape(data, (np.array(self.data).shape[0], -1, data.shape[-1]))
        columns = ~ ((self.output_properties[1:] == 'mass') | (self.output_properties[1:] == 'xu') | (
                self.output_properties[1:] == 'yu') | (self.output_properties[1:] == 'zu'))
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
        print("Preparing input\t", end="")
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

        train_features = Pool().map(self._construct_features, vec_packed_t_data)
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

        self.feature_matrix_t = feature_matrix_t
        self.target_vector = target_vector

        print(np.round(time.time() - t_, 2), "s")

    def regress(self, method='simple'):
        feature_matrix = []
        target_vector = []
        if method == 'simple':
            feature_matrix = np.swapaxes(self.feature_matrix_t, 0, 1)
            feature_matrix = np.reshape(feature_matrix, (np.size(feature_matrix, axis=0),
                                                         np.size(feature_matrix, axis=1) * np.size(
                                                             feature_matrix, axis=2)))
            target_vector = np.ravel(self.target_vector)

        if method == "bayesian":
            feature_matrix = np.swapaxes(self.feature_matrix_t, 0, 1)
            feature_matrix = np.sum(feature_matrix, axis=1)
            target_vector = np.sum(self.target_vector, axis=0)

        projection = np.matmul(feature_matrix, target_vector)
        norm = np.matmul(feature_matrix, feature_matrix.T)
        norm_inverse = np.linalg.inv(norm)
        self.weights = np.matmul(norm_inverse, projection)

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
            Y.append(y / 4.184e-4) # ` unit conversion from kcal
        return Y
