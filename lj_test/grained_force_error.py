import numpy as np


class VF_grain:

    def __init__(self, outfile_path="", res=100, timestep=0.5):
        self.outfile_path = outfile_path
        self.res = res
        self.timestep = timestep
        self.r_target = np.empty(0)
        self.v_target = np.empty(0)
        self.a_target = np.empty(0)
        self.init_targets()

        self.v_derived = {}
        self.a_derived = {}
        self.v_derived_error = {}
        self.a_derived_error = {}

    def read_data(self):
        data = []
        with open(self.outfile_path) as f:
            l = len(f.readlines())
        with open(self.outfile_path) as f:
            for i in range(l):
                line = f.readline().split()
                if len(line) == 2:
                    if line[1] == "TIMESTEP":
                        data.append([])
                elif len(line) == 11:
                    data[-1].append(np.array(line).astype(float)[1:])
        return np.array(data)

    def init_targets(self):
        data = self.read_data()
        m = data[0, 0, 0]
        r_target = data[:, :, 1:4]
        v_target = data[:, :, 4:7]
        a_target = data[:, :, 7:10] * 4.184e-4 / m
        self.r_target, self.v_target, self.a_target = r_target, v_target, a_target

    def grain_velocity_force(self):
        r = self.r_target[::self.res]

        timestep = self.timestep * self.res

        v_derived = (r[1:] - r[:-1]) / timestep
        a_derived = (v_derived[1:] - v_derived[:-1]) / timestep
        # derived_a = (r[2:] - 2 * r[1:-1] + r[:-2]) / timestep ** 2

        self.v_derived[self.res] = v_derived
        self.a_derived[self.res] = a_derived

        self.error_velocity_force()

    def error_velocity_force(self):
        target_v = self.v_target[::self.res, :, :][:-1]
        target_a = self.a_target[::self.res, :, :][1:-1]
        target_v_norm = np.sqrt(np.sum(target_v ** 2, axis=-1))
        target_a_norm = np.sqrt(np.sum(target_a ** 2, axis=-1))

        error_v = np.sqrt(np.sum((target_v - self.v_derived[self.res]) ** 2, axis=-1))
        error_a = np.sqrt(np.sum((target_a - self.a_derived[self.res]) ** 2, axis=-1))
        norm_error_v = error_v / target_v_norm
        norm_error_a = error_a / target_a_norm

        self.v_derived_error[self.res] = np.mean(norm_error_v)
        self.a_derived_error[self.res] = np.mean(norm_error_a)
