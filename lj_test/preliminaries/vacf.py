import numpy as np


class VACF:

    def __init__(self, outfile_path = ""):
        self.outfile_path = outfile_path

    def read_data(self):
        data = []
        with open(self.outfile_path) as f:
            l = len(f.readlines())
        with open(self.outfile_path) as f:
            for i in range(l):
                line = f.readline().split()
                if len(line) == 2:
                    data.append([])
                else:
                    data[-1].append(np.array(line).astype(float)[-1])

        return np.array(data)
