import numpy as np
import time
import os

class DataProcessing:

    def __init__(self, outfile_path, timestep, cutoff, basis, basis_parameters, steps_between_points=1, steps_used_in_fit=100):
        self.t = steps_between_points
        self.n = steps_used_in_fit
        self.outfile_path = outfile_path
        self.timestep = timestep
        self.cutoff = cutoff
        self.basis = basis
        self.basis_params = basis_parameters

        self.atom_types = []
        self.box_dimensions = []
        self.data = []
        self.r = []
        self.ru = []
        self.a = []
        self.f = []
        self.read_data()

        self.feature_matrix_t = []
        self.target_vector = []
        self.weights = []

    def read_data(self):
        number_of_output_properties = 11
        print("Reading data\t",end="")
        t_ = time.time()

        atom_types_dict = {}
        atom_types = []
        t = -1
        id = 0
        with open(self.outfile_path) as f:
            while t < 1:
                line = f.readline().split()
                if len(line) > 1:
                    if line[1] == "TIMESTEP":
                        t += 1
                if len(line) == number_of_output_properties:
                    mass = line[1]
                    if mass not in list(atom_types_dict.keys()):
                        id += 1
                        atom_types_dict[mass] = id
                        atom_types.append(atom_types_dict[mass])
                    else:
                        atom_types.append(atom_types_dict[mass])
        self.atom_types = atom_types

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
                        for j in range(3):
                            line = f.readline().split()
                            dimensions.append(float(line[1]) - float(line[0]))
                        i += 3
                        self.box_dimensions.append(dimensions)
                elif len(line) == number_of_output_properties and t % self.t == 0:
                    data[-1].append(np.array(line).astype(float)[1:])

        self.data = np.array(data)
        self.r = self.data[:, :, 1:4]
        self.ru = self.data[:, :, 4:7]
        self.f = self.data[:, :, 7:10][1:-1]
        v = (self.r[1:] - self.r[:-1]) / self.timestep
        self.a = (v[1:] - v[:-1]) / self.timestep
        print(np.round(time.time(), 2) - t_,"s")

    def prepare_training_data(self):
        print("Preparing input\t",end="")
        t_ = time.time()
        np.seterr(all='ignore')
        x_train = []
        y_train = []
        z_train = []

        for t in range(len(self.data)):
            lx, ly, lz = self.box_dimensions[t][0], self.box_dimensions[t][1], self.box_dimensions[t][2]
            relative_pos_x = d_x = -np.subtract.outer(self.r[t, :, 0], self.r[t, :, 0])
            relative_pos_y = d_y = -np.subtract.outer(self.r[t, :, 1], self.r[t, :, 1])
            relative_pos_z = d_z = -np.subtract.outer(self.r[t, :, 2], self.r[t, :, 2])
            pbc_relative_pos_x = d_pbc_x = np.where(np.abs(d_x) >= 0.5 * lx, d_x - np.sign(d_x) * lx, relative_pos_x)
            pbc_relative_pos_y = d_pbc_y = np.where(np.abs(d_y) >= 0.5 * ly, d_y - np.sign(d_y) * ly, relative_pos_y)
            pbc_relative_pos_z = d_pbc_z = np.where(np.abs(d_z) >= 0.5 * lz, d_z - np.sign(d_z) * lz, relative_pos_z)
            pbc_dist = np.sqrt(d_pbc_x ** 2 + d_pbc_y ** 2 + d_pbc_z ** 2)

            x_train.append([])
            y_train.append([])
            z_train.append([])
            for p in self.basis_params:
                force_p = self.basis(pbc_dist, p)
                force_p = np.nan_to_num(force_p, posinf=0, neginf=0, nan=0)
                force_p = np.where(pbc_dist > self.cutoff, 0, force_p)

                force_p_x = np.nan_to_num(force_p * pbc_relative_pos_x / pbc_dist, posinf=0, neginf=0, nan=0)
                force_p_y = np.nan_to_num(force_p * pbc_relative_pos_y / pbc_dist, posinf=0, neginf=0, nan=0)
                force_p_z = np.nan_to_num(force_p * pbc_relative_pos_z / pbc_dist, posinf=0, neginf=0, nan=0)

                x_train[-1].append(np.sum(force_p_x, axis=0))
                y_train[-1].append(np.sum(force_p_y, axis=0))
                z_train[-1].append(np.sum(force_p_z, axis=0))

        m = self.data[0][0][0]
        target_vector = np.array(self.a) * m
        target_vector = np.swapaxes(target_vector, 1, 2)
        target_vector = np.reshape(target_vector, (np.size(self.a, axis=0),
                                                   np.size(self.a, axis=2) * np.size(self.a, axis=1)))

        feature_matrix_t = np.concatenate((x_train, y_train, z_train), axis=2)[1:-1]

        self.feature_matrix_t = feature_matrix_t
        self.target_vector = target_vector
        print(np.round(time.time() - t_, 2), "s")

    def regress(self, method='simple'):
        t = time.time()
        feature_matrix_t = []
        target_vector = []
        if method == 'simple':
            feature_matrix_t = np.swapaxes(self.feature_matrix_t, 0, 1)
            feature_matrix_t = np.reshape(feature_matrix_t, (np.size(feature_matrix_t, axis=0),
                                                             np.size(feature_matrix_t, axis=1) * np.size(
                                                                 feature_matrix_t, axis=2)))
            target_vector = np.ravel(self.target_vector)

        if method == "bayesian":
            feature_matrix_t = np.swapaxes(self.feature_matrix_t, 0, 1)
            feature_matrix_t = np.sum(feature_matrix_t, axis=1)
            target_vector = np.sum(self.target_vector, axis=0)


        projection = np.matmul(feature_matrix_t, target_vector)
        norm = np.matmul(feature_matrix_t, feature_matrix_t.T)
        t = time.time()
        norm_inverse = np.linalg.inv(norm)
        self.weights = np.matmul(norm_inverse, projection)

    def predict(self, x):
        if type(x) == float:
            y = 0
        else:
            y = np.zeros(len(x))
        for i in range(len(self.basis_params)):
            y = y + self.weights[i] * self.basis(x, self.basis_params[i])
        return y / 4.184e-4 #` unit conversion from kcal
