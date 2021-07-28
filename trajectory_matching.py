import itertools

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from misc import plot_1component


class TrajectoryMatching:

    def __init__(self, outfile_path, basis, basis_parameters, simulation_timestep=1., cutoff=20, every_n_from_output=1,
                 timesteps_in_fit=100, system_style='atomic', op_sys='Linux', cg_instructions=None):

        if system_style not in ["atomic", "molecular"]:
            raise ValueError("'system_style' most be either 'atomic' or 'molecular'")
        else:
            self.system_style = system_style
        self.op_sys = op_sys

        self.t = every_n_from_output
        self.n = timesteps_in_fit + 2  # first and last position does not correspond to acceleration
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
        self.ru = []
        self.v = []
        self.a = []
        self.f = []
        self.cons_force_table_dict = {}

        if cg_instructions is None:
            self.cg_instructions = {}
        else:
            self.cg_instructions = cg_instructions
        self.bond_length = 0.
        self.bonded_pairs = []
        self.feature_matrix = []
        self.target_vector = []
        self.weights = []
        self.bonded_weight = 0.

        self.read_data()

    def read_data(self):
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

        l = sum(1 for _ in open(self.outfile_path, 'rb'))

        data_append = 0
        box_append = 0

        with open(self.outfile_path) as f:
            t = -1
            pbar = tqdm(total=self.n, desc="loading")
            data_t = []
            for i in range(l):
                line = f.readline().split()
                if len(line) == 0:
                    continue
                if len(line) == 2:
                    if line[1] == "TIMESTEP":
                        t += 1
                        if (t - 1) % self.t == 0:
                            data_append += 1
                            if self.system_style == "molecular" and len(data_t) > 0:
                                data_t = np.array(data_t)
                                mol_ids = np.unique(data_t[:, int(np.where(self.output_properties[1:] == 'mol')[0][0])])
                                reduced_data = []
                                for id in mol_ids:
                                    data_mol = self.reduce_to_centre_of_mass(data_t, id, len(reduced_data))
                                    for data_ in data_mol:
                                        reduced_data.append(data_)
                                self.data.append(np.array(reduced_data))
                                pbar.update(1)
                            else:
                                if len(data_t) > 0:
                                    self.data.append(np.array(data_t))
                                    pbar.update(1)
                            if len(self.data) == self.n:
                                break
                            data_t = []
                elif line[0] != 'ITEM:' and len(line) == len(output_properties) and t % self.t == 0:
                    data_t.append(np.array(line).astype(float)[1:])
                elif len(line) == 6 and t % self.t == 0:
                    if line[1] == "BOX":
                        box_append += 1
                        dimensions = []
                        self.box_lo.append([])
                        for j in range(3):
                            line = f.readline().split()
                            dimensions.append(float(line[1]) - float(line[0]))
                            self.box_lo[-1].append(float(line[0]))
                        i += 3
                        self.box_dimensions.append(dimensions)
        pbar.close()

        self.box_dimensions = np.array(self.box_dimensions)[:len(self.data)]
        self.box_lo = np.array(self.box_lo)[:len(self.data)]
        self.data = np.array(self.data)


        if self.system_style == "molecular" and len(self.cg_instructions.keys()) > 0:
            bonded_pairs = np.zeros((self.data.shape[1], self.data.shape[1]))
            for pair in self.bonded_pairs:
                bonded_pairs[pair[0], pair[1]] = 1
            self.bonded_pairs = bonded_pairs
            self.calculate_eq_bond_length()

        if self.system_style == "atomic":
            f_columns = np.nonzero(~ ((self.output_properties[1:] == 'fx') | (self.output_properties[1:] == 'fy') |
                                      (self.output_properties[1:] == 'fz')))
            self.f = np.array(np.delete(self.data, f_columns, axis=2))

        if self.system_style == "molecular":

            ru = self.data[:, :, 1:4]
            self.ru = ru

            shift_to_centre = np.repeat([np.array(self.box_lo)], np.array(self.data).shape[1], axis=0)
            shift_to_centre = np.swapaxes(shift_to_centre, 0, 1)
            shift_to_unit_cell = np.repeat([np.array(self.box_dimensions)], np.array(self.data).shape[1], axis=0)
            shift_to_unit_cell = np.swapaxes(shift_to_unit_cell, 0, 1)

            self.r = ru - shift_to_centre
            self.r = np.floor(self.r / shift_to_unit_cell) * shift_to_unit_cell
            self.r = ru - self.r

        elif self.system_style == "atomic":
            self.ru = self.data[:, :, 4:7]
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

    def calculate_eq_bond_length(self):
        lengths = []
        for pair in self.bonded_pairs[::2]:
            pair_id = np.nonzero(pair == 1.)
            if not len(pair_id[0]) == 0:
                id1 = int(pair_id[0])
                id2 = id1 - 1
            else:
                continue
            pos1 = self.data[:, id1, 1:]
            pos2 = self.data[:, id2, 1:]
            length = np.sum((pos1 - pos2) ** 2, axis=1)
            lengths.append(np.sqrt(length))
        self.bond_length = np.average(lengths)

    def reduce_to_centre_of_mass(self, data, mol_id, list_l):
        data = data[data[:, 1] == mol_id]
        data = np.reshape(data, (-1, data.shape[-1]))
        columns = np.nonzero(~ ((self.output_properties[1:] == 'mass') | (self.output_properties[1:] == 'xu') | (
                self.output_properties[1:] == 'yu') | (self.output_properties[1:] == 'zu')))
        data = np.delete(data, columns, axis=1)

        if int(mol_id) not in list(self.cg_instructions.keys()):
            m = np.sum(data, axis=0)[0]
            data = np.swapaxes(data, 0, 1)
            data[1:] = data[1:] * data[0] / m
            data = np.swapaxes(data, 0, 1)
            data = np.sum(data, axis=0)
            data = data[np.newaxis, :]
        else:
            self.bonded_pairs.append([list_l, list_l + 1])
            self.bonded_pairs.append([list_l + 1, list_l])
            data_list = []
            duplicates = []
            for id in self.cg_instructions[mol_id][0]:
                if id in self.cg_instructions[mol_id][1]:
                    duplicates.append(id)
            duplicates = np.array(duplicates)
            data[duplicates, 0] /= 2.
            for atom_ids in self.cg_instructions[mol_id]:
                atom_ids = np.array(atom_ids)
                m = np.sum(data[atom_ids], axis=0)[0]
                data_ = data[atom_ids, :]
                data_ = np.swapaxes(data_, 0, 1)
                data_[1:] = data_[1:] * data_[0] / m
                data_ = np.swapaxes(data_, 0, 1)
                data_ = np.sum(data_, axis=0)
                data_list.append(data_)
            data = data_list
        return data

    def _construct_features(self, r_t, box_dimensions):
        np.seterr(all='ignore')

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
                if len(self.cg_instructions.keys()) > 0:
                    force_p = np.where(self.bonded_pairs == 1., 0, force_p)
                force_p_x = np.nan_to_num(force_p * pbc_relative_pos_x / pbc_dist, posinf=0, neginf=0, nan=0)
                force_p_y = np.nan_to_num(force_p * pbc_relative_pos_y / pbc_dist, posinf=0, neginf=0, nan=0)
                force_p_z = np.nan_to_num(force_p * pbc_relative_pos_z / pbc_dist, posinf=0, neginf=0, nan=0)

                x_train.append(np.sum(force_p_x, axis=0))
                y_train.append(np.sum(force_p_y, axis=0))
                z_train.append(np.sum(force_p_z, axis=0))

        # TODO: generalise so that more than one type of monomer can be cg'd to double-beads
        if len(self.cg_instructions.keys()) > 0:
            force_bonded = np.abs(pbc_dist) - self.bond_length
            force_bonded = np.nan_to_num(force_bonded, posinf=0, neginf=0, nan=0)
            force_bonded = np.where(self.bonded_pairs == 0., 0, force_bonded)

            force_p_x = np.nan_to_num(force_bonded * pbc_relative_pos_x / pbc_dist, posinf=0, neginf=0, nan=0)
            force_p_y = np.nan_to_num(force_bonded * pbc_relative_pos_y / pbc_dist, posinf=0, neginf=0, nan=0)
            force_p_z = np.nan_to_num(force_bonded * pbc_relative_pos_z / pbc_dist, posinf=0, neginf=0, nan=0)

            x_train.append(np.sum(force_p_x, axis=0))
            y_train.append(np.sum(force_p_y, axis=0))
            z_train.append(np.sum(force_p_z, axis=0))

        return [x_train, y_train, z_train]

    def prepare_training_data(self, t):
        train_features = self._construct_features(self.r[t + 1], self.box_dimensions[t + 1])  # t + 1 'r' => t 'a'
        x_train = train_features[0]
        y_train = train_features[1]
        z_train = train_features[2]
        feature_matrix_t = np.concatenate((x_train, y_train, z_train), axis=1)

        target_vector = np.swapaxes(self.a[t], 0, 1)
        target_vector = np.reshape(target_vector, -1)

        m = np.ravel(list([self.data[0, :, 0]]) * 3)
        target_vector = target_vector * m / 4.184e-4

        return feature_matrix_t, target_vector

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

    def _fit_core(self, t, method):
        feature_matrix_t, target_vector_t = self.prepare_training_data(t)
        if method == 'nve' or method == 'NVE':
            norm = np.matmul(feature_matrix_t, feature_matrix_t.T)
            projection = np.matmul(feature_matrix_t, target_vector_t)
        else:
            b_matrix = self.b_matrix(t)
            b_matrix_t_pinv = np.linalg.pinv(b_matrix)
            norm = np.matmul(b_matrix_t_pinv, feature_matrix_t.T)
            norm = np.matmul(feature_matrix_t, norm)
            projection = np.matmul(b_matrix_t_pinv, target_vector_t)
            projection = np.matmul(feature_matrix_t, projection)

        return [norm, projection]

    def fit(self, method=''):
        projections = []
        norms = []
        pbar = tqdm(total=len(self.a), desc="fitting")
        for t in range(len(self.r) - 2):
            norm_projection = self._fit_core(t, method)
            norms.append(np.array(norm_projection[0]))
            projections.append(np.array(norm_projection[1]))
            pbar.update(1)
        pbar.close()

        projections, norms = np.array(projections), np.array(norms)
        projection = np.sum(projections, axis=0)
        norm = np.sum(norms, axis=0)
        norm_inverse = np.linalg.inv(norm)
        self.weights = np.matmul(norm_inverse, projection)  # unit conversion to 'LAMMPS real'

        if len(self.cg_instructions.keys()) > 0:
            self.bonded_weight = self.weights[-1]
            self.weights = self.weights[:-1]

        if self.bond_length > 0:
            print("equilibrium bond length: ", self.bond_length)
            print("bonded weight: ", polymer_tm.bonded_weight)

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
            weights = np.append(self.weights, self.bonded_weight)
            f_c = np.matmul(self.feature_matrix[t].T, np.array(weights))
            f_diff = (self.target_vector[t] - f_c)
            g_0 = np.matmul(b_matrix_pinv, f_diff)
            gamma_0 += np.matmul(f_diff, g_0)

        gamma_0, gamma_2 = -gamma_0 / n, gamma_2 / n
        gamma_1 = 3 * 2 * self.r.shape[1] * k * T / (self.timestep)

        return np.roots([gamma_2, gamma_1, gamma_0])[1]

    def best_subset(self, k_list, x, force_index, plot=False):
        original_weights = np.array(self.weights).copy()
        original_params = np.array(self.basis_params).copy()
        y_target = self.predict(x)[force_index]
        RSS = {}
        weights = {}
        if type(k_list) == int:
            k_list = [k_list]
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
                RSS[tuple(params)] = np.sqrt(np.sum((self.predict(x, single=True)[0] - y_target) ** 2)) / float(
                    len(x))
        RSS = dict(sorted(RSS.items(), key=lambda item: item[1]))
        key = list(RSS.keys())[0]
        self.weights = weights[key]
        self.basis_params = np.array(key)
        if plot:
            y_fit_ = self.predict(x, single=True)[0]
            if len(y_target) == len(x):
                plot_1component(x, y_fit_, y_target, labels=["reduced parameter set", "TM fit"])
            else:
                plot_1component(x, y_fit_, labels=["TM fit"])
        self.basis_params = original_params
        self.weights = original_weights
        return np.array(key), weights[key]

    def predict(self, x, single=False):
        number_of_type_pairs = len(self.unique_type_pairs)
        if single:
            number_of_type_pairs = 1
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
        return Y

    def predict_energy(self, x, force=None):
        number_of_type_pairs = len(self.unique_type_pairs)
        if force is None:
            Y_force = self.predict(x)
        else:
            Y_force = [force]
        Y = []
        for i in range(number_of_type_pairs):
            f = Y_force[i]
            e_ = []
            for i in range(len(f) - 1, -1, -1):
                e_.append(np.trapz(f[i:], x[i:]))
            e_ = e_[::-1]
            Y.append(e_)
            if force is not None:
                break
        return Y

    def radial_distirbution_function(self, dr, pair_type=None, bonded_pairs=True, only_bonded_pairs=False, plot=True):
        if pair_type is None:
            pair_type = self.unique_type_pairs[0]

        pbc_dist_list = []
        r = self.r
        for t in range(len(r)):
            lx, ly, lz = self.box_dimensions[t][0], self.box_dimensions[t][1], \
                         self.box_dimensions[t][2]
            relative_pos_x = d_x = -np.subtract.outer(r[t][:, 0], r[t][:, 0])
            relative_pos_y = d_y = -np.subtract.outer(r[t][:, 1], r[t][:, 1])
            relative_pos_z = d_z = -np.subtract.outer(r[t][:, 2], r[t][:, 2])
            pbc_relative_pos_x = np.where(np.abs(d_x) >= 0.5 * lx, d_x - np.sign(d_x) * lx, relative_pos_x)
            pbc_relative_pos_y = np.where(np.abs(d_y) >= 0.5 * ly, d_y - np.sign(d_y) * ly, relative_pos_y)
            pbc_relative_pos_z = np.where(np.abs(d_z) >= 0.5 * lz, d_z - np.sign(d_z) * lz, relative_pos_z)
            pbc_dist = np.sqrt(pbc_relative_pos_x ** 2 + pbc_relative_pos_y ** 2 + pbc_relative_pos_z ** 2)

            pbc_dist = np.where(self.atom_type_pairs == pair_type, pbc_dist, 0)

            if not bonded_pairs:
                pbc_dist = np.where(self.bonded_pairs == 1., 0., pbc_dist)
            elif only_bonded_pairs:
                pbc_dist = np.where(self.bonded_pairs == 0., 0., pbc_dist)

            pbc_dist_list.append(pbc_dist)

        pbc_dist_list = np.array(pbc_dist_list).reshape(-1)
        pbc_dist_list = pbc_dist_list[pbc_dist_list != 0.]
        pbc_dist_list = pbc_dist_list[pbc_dist_list < np.mean(self.box_dimensions) / 2]
        bin_list = np.floor(pbc_dist_list / dr).astype(int)

        m = max(bin_list)
        count, bins = np.histogram(bin_list, bins=int(m), range=[0, m])
        bins = bins[:-1] * dr + dr / 2
        count, bins = count[:-1], bins[:-1]
        count = count / (4 * np.pi * bins ** 2 * dr)
        count /= (len(r) * len(r[0]))
        count /= (len(r[0]) / np.mean(self.box_dimensions) ** 3)

        count *= (self.atom_type_pairs.size / (self.atom_type_pairs[self.atom_type_pairs == pair_type]).size)

        if plot:
            plt.axhline(1, ls='--', color="xkcd:silver")
            plt.plot(bins, count)
            plt.xlabel(r"$r ({\AA})$")
            plt.ylabel(r"$g(r)")
            plt.title("Radial distirbution function")

        return bins, count

    def write_table(self, filename, key, x, e, f):
        file_ = open(filename, 'w')
        file_.write(f"{key}\n")
        file_.write(f"N {len(x)}\tRSQ {x[0]} {x[-1]}\n\n")
        for ix, x_ in enumerate(x):
            file_.write(f"{ix + 1} {x_} {e[ix]} {f[ix]}\n")
        file_.close()

def basis_function(x, p):
    return np.sign(x) * np.abs(x) ** p


if __name__ == "__main__":

    T = 350
    PATH = "/home/markjenei/trajectory-matching/epoxy/"
    outfile = PATH + "epoxy.out"


    run = 1
    steps_between_points = 1
    configurations = 20
    params = np.array(range(-1, -23, -2))

    cg_ids_1 = [0] + list(range(1, 25))
    cg_ids_2 = [0] + list(range(25, 49))

    cg_instructions = {}
    for i in range(1, 401):
        cg_instructions[i] = [cg_ids_1, cg_ids_2]

    polymer_tm = TrajectoryMatching(outfile_path=outfile, basis=basis_function, basis_parameters=params,
                                    simulation_timestep=0.5, cutoff=50, system_style='molecular',
                                    every_n_from_output=steps_between_points, timesteps_in_fit=configurations,
                                    cg_instructions=cg_instructions)

    polymer_tm.fit()
    np.save(PATH + f"epoxy_{run}.npy", polymer_tm.weights)

    x = np.linspace(7.0, 40, 10000)
    y_fit_long = polymer_tm.predict(x)[0]
    plot_1component(x, y_fit_long, output_path=PATH + f"epoxy_{run}_a.png")
    x = np.linspace(2.5, 50, 10000)
    y_fit_long = polymer_tm.predict(x)[0]
    plot_1component(x, y_fit_long, output_path=PATH + f"epoxy_{run}_b.png")