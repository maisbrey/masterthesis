import numpy as np
import galois
from .utils import *

class QuantumCode:
    def __init__(self):
        self.GF4 = galois.GF(4)

    def read_alist(self, file_path):
        file_lines = open(file_path, "r").readlines()

        def extract_ints(file_lines, line):
            list_buffer = file_lines[line].replace("\n", "").split(" ")
            if "" in list_buffer:
                list_buffer.remove("")
            return np.array(list(map(int, list_buffer)))

        self.mbin, self.nbin = extract_ints(file_lines, 0)

        self.Mbin_deg = extract_ints(file_lines, 2)
        self.Nbin_deg = extract_ints(file_lines, 3)

        self.Nbin = (-1) * np.ones((self.mbin, np.max(self.Mbin_deg)), dtype=np.uint16)
        self.Mbin = (-1) * np.ones((self.nbin, np.max(self.Nbin_deg)), dtype=np.uint16)

        for row in np.arange(self.mbin):
            self.Nbin[row, :self.Mbin_deg[row]] = extract_ints(file_lines, 4 + row)
        for column in np.arange(self.nbin):
            self.Mbin[column, :self.Nbin_deg[column]] = extract_ints(file_lines, 4 + self.mbin + column)


    def bin_to_decomp_index_based(self):
        self.Mx = self.Mbin[:int(self.nbin / 2)]
        self.Mz = self.Mbin[int(self.nbin / 2):]

        self.Nx = np.where(self.Nbin < int(self.nbin / 2), self.Nbin, -1)
        self.Nz = np.where(self.Nbin >= int(self.nbin / 2), self.Nbin, -1)
        positions = np.argwhere(self.Nz != -1).T
        shift_rows = np.unique(positions[0])
        shift_vector = -np.array([positions[1][positions[0] == ctr][0] for ctr in shift_rows])
        column_indices = (np.arange(self.Nz.shape[1]) - shift_vector[:, np.newaxis]) % self.mbin
        self.Nz[shift_rows] = self.Nz[shift_rows[:, np.newaxis], column_indices]
        self.Nz[self.Nz != -1] -= int(self.nbin / 2)

    def decomp_to_quat_index_based(self, Ax, Az):
        matr_stacked = np.hstack((Ax, Az))

        reorder_index = np.arange(Ax.shape[0])[:, np.newaxis], np.argsort(matr_stacked)
        matr_stacked_reordered = matr_stacked[reorder_index]

        matr_pauli_lookup_reordered = np.hstack(((Ax != -1).astype(np.uint8), 2 * (Az != -1).astype(np.uint8)))
        matr_pauli_lookup_reordered = matr_pauli_lookup_reordered[reorder_index]

        matr = (-1) * np.ones_like(matr_stacked)
        matr_pauli_lookup = (-1) * np.ones_like(matr_stacked)

        for node, (R, Q) in enumerate(zip(matr_stacked_reordered, matr_pauli_lookup_reordered)):
            neighbor_indices, counts = np.unique(R[R != -1], return_counts=True)
            pauli_entry = np.zeros_like(neighbor_indices)
            pauli_entry[counts == 1] = Q[R != -1][np.isin(R[R != -1], neighbor_indices[counts == 1])]
            pauli_entry[counts > 1] = 3
            matr[node, :neighbor_indices.size] = neighbor_indices
            matr_pauli_lookup[node, :neighbor_indices.size] = pauli_entry

        b = np.min(np.sum(matr == -1, axis=1))

        return matr[:, :matr.shape[1] - b], matr_pauli_lookup[:, :matr.shape[1] - b]

    def set_quat_index_based(self):
        self.Mr, self.M_pauli_lookup = self.decomp_to_quat_index_based(self.Mx, self.Mz)
        self.Nr, self.N_pauli_lookup = self.decomp_to_quat_index_based(self.Nx, self.Nz)

        self.N_deg = np.sum(self.Mr != -1, axis=1)
        self.M_deg = np.sum(self.Nr != -1, axis=1)

        self.m = self.M_deg.shape[0]
        self.n = self.N_deg.shape[0]

        #self.column_matrices()
        self.Mc, self.Nc = self.get_column_matrices(self.Mr, self.Nr)

    def set_CSS_index_based(self):
        a = np.argwhere(np.all(self.Nx == -1, axis=1)).flatten()
        if a.size == 0:
            self.css_bool = False
        else:
            self.css_bool = True
            self.mx = a[0]
            self.mz = self.m - self.mx
            self.Mxr_CSS = self.Mx
            self.Mzr_CSS = self.Mz - self.mx * (self.Mz != -1)
            self.Nxr_CSS = self.Nx[:self.mx]
            self.Nzr_CSS = self.Nz[self.mx:]
            self.Nx_CSS_deg = np.sum(self.Mxr_CSS != -1, axis = 1)
            self.Nz_CSS_deg = np.sum(self.Mzr_CSS != -1, axis = 1)
            self.Mx_CSS_deg = np.sum(self.Nxr_CSS != -1, axis = 1)
            self.Mz_CSS_deg = np.sum(self.Nzr_CSS != -1, axis = 1)

            self.Mxc_CSS, self.Nxc_CSS = self.get_column_matrices(self.Mxr_CSS, self.Nxr_CSS)
            self.Mzc_CSS, self.Nzc_CSS = self.get_column_matrices(self.Mzr_CSS, self.Nzr_CSS)

    ####################################################################################################################

    def CSS_to_decomp(self):
        self.Hx = np.vstack((self.Hx_CSS, np.zeros_like(self.Hz_CSS)))
        self.Hz = np.vstack((np.zeros_like(self.Hx_CSS), self.Hz_CSS))

    def decomp_to_CSS(self):
        self.Hx_CSS = self.Hx[:self.mx]
        self.Hz_CSS = self.Hz[self.mx:]

    def quat_to_decomp(self):
        self.Hx = np.isin(self.H, [1, 3]).astype(np.uint8)
        self.Hz = np.isin(self.H, [2, 3]).astype(np.uint8)

    def decomp_to_quat(self):
        self.H = self.GF4(self.Hx + 2 * self.Hz)

    def decomp_to_bin(self):
        self.Hbin = np.hstack((self.Hx, self.Hz))

        self.Mbin, self.Nbin_deg, _ = self.get_node_matrix(self.Hbin.T)
        self.Nbin, self.Mbin_deg, _ = self.get_node_matrix(self.Hbin)

        self.mbin, self.nbin = self.Hbin.shape

        self.N_rank_check = [sum([2 ** int(VN) for VN in self.Nbin[CN, :self.Mbin_deg[CN]]]) for CN in np.arange(self.mbin)]
        self.S_rank = self.gf2_rank(self.N_rank_check)

    def bin_to_decomp(self):
        self.Hx = self.Hbin[:, :int(self.nbin / 2)]
        self.Hz = self.Hbin[:, int(self.nbin / 2):]

    ####################################################################################################################

    def get_node_matrix(self, H):
        indices = np.argwhere(H != 0).T
        degs = (H != 0).T.sum(axis=0)
        node_matrix = (-1) * np.ones((H.shape[0], np.max(degs)), dtype=np.uint16)
        for row in np.arange(indices[0, -1] + 1):
            node_matrix[row, :np.sum(indices[0] == row)] = indices[1][indices[0] == row]
        return node_matrix, degs, np.vstack((indices[0],
                                             np.tile(np.arange(np.max(degs))[np.newaxis, :],
                                                     reps=(indices[0, -1] + 1, 1))[node_matrix != -1],
                                             indices[1]))

    def H_from_index_based(self):
        self.H = np.zeros((self.m, self.n), dtype=np.uint8)
        for row, columns in enumerate([self.Nr[column, :self.M_deg[column]] for column in np.arange(self.m)]):
            self.H[row, columns] = self.N_pauli_lookup[row][:self.M_deg[row]].astype(np.uint8)

    def index_based_from_quat(self):
        self.Mr, self.N_deg, _ = self.get_node_matrix(self.H.T)
        self.Nr, self.M_deg, _ = self.get_node_matrix(self.H)

        self.column_matrices()

    def column_matrices(self):
        self.Mc = np.zeros_like(self.Mr)
        visited = np.zeros(self.m)
        for VN, CNs in enumerate(self.Mr):
            self.Mc[VN] = visited[CNs]
            visited[CNs[CNs != -1]] += 1

        self.Mc[self.Mr == -1] = -1

        self.Nc = np.zeros_like(self.Nr)
        visited = np.zeros(self.n)
        for CN, VNs in enumerate(self.Nr):
            self.Nc[CN] = visited[VNs]
            visited[VNs[VNs != -1]] += 1

        self.Nc[self.Nr == -1] = -1

    def get_column_matrices(self, Ar, Br):
        Ac = np.zeros_like(Ar)
        visited = np.zeros(Br.shape[0])
        for VN, CNs in enumerate(Ar):
            Ac[VN] = visited[CNs]
            visited[CNs[CNs != -1]] += 1
        Ac[Ar == -1] = -1

        Bc = np.zeros_like(Br)
        visited = np.zeros(Ar.shape[0])
        for CN, VNs in enumerate(Br):
            Bc[CN] = visited[VNs]
            visited[VNs[VNs != -1]] += 1
        Bc[Br == -1] = -1

        return Ac, Bc

    def column_matrices_CSS(self):
        self.Mxc_CSS = np.zeros_like(self.Mxr_CSS)
        visited = np.zeros(self.mx)
        for VN, CNs in enumerate(self.Mxr_CSS):
            self.Mxc_CSS[VN] = visited[CNs]
            visited[CNs[CNs != -1]] += 1

        self.Mxc_CSS[self.Mxr_CSS == -1] = -1

        self.Nxc_CSS = np.zeros_like(self.Nxr_CSS)
        visited = np.zeros(self.n)
        for CN, VNs in enumerate(self.Nxr_CSS):
            self.Nxc_CSS[CN] = visited[VNs]
            visited[VNs[VNs != -1]] += 1

        self.Nxc_CSS[self.Nxr_CSS == -1] = -1

        self.Mzc_CSS = np.zeros_like(self.Mzr_CSS)
        visited = np.zeros(self.mz)
        for VN, CNs in enumerate(self.Mxr_CSS):
            self.Mzc_CSS[VN] = visited[CNs]
            visited[CNs[CNs != -1]] += 1

        self.Mzc_CSS[self.Mzr_CSS == -1] = -1

        self.Nzc_CSS = np.zeros_like(self.Nzr_CSS)
        visited = np.zeros(self.n)
        for CN, VNs in enumerate(self.Nzr_CSS):
            self.Nzc_CSS[CN] = visited[VNs]
            visited[VNs[VNs != -1]] += 1

        self.Nzc_CSS[self.Nzr_CSS == -1] = -1

    def write_alist(self, file_path):
        file = open(file_path, 'w')
        file.write(" ".join([str(entry) for entry in self.Hbin.shape]) + "\n")
        file.write(str(np.max(self.Mbin_deg)) + " " + str(np.max(self.Nbin_deg)) + "\n")
        file.write(" ".join(self.Mbin_deg.astype(str)) + "\n")
        file.write(" ".join(self.Nbin_deg.astype(str)) + "\n")

        for m in np.arange(self.Hbin.shape[0]):
            file.write(" ".join(self.Nbin[m, :self.Mbin_deg[m]].astype(str)) + "\n")

        for n in np.arange(self.Hbin.shape[1]):
            file.write(" ".join(self.Mbin[n, :self.Nbin_deg[n]].astype(str)) + "\n")

        file.close()

    def gf2_rank(self, rows_input):
        rows = rows_input.copy()
        rank = 0
        while rows:
            pivot_row = rows.pop()
            if pivot_row:
                rank += 1
                lsb = pivot_row & -pivot_row
                for index, row in enumerate(rows):
                    if row & lsb:
                        rows[index] = row ^ pivot_row
        return rank

########################################################################################################################

    def read_mode(self, file_path, css_bool=True):
        self.read_alist(file_path)          # Mbin, Mbin_deg, Nbin, Nbin_deg, mbin, nbin
        self.bin_to_decomp_index_based()    # Mx, Mz, Nx, Nz

        self.set_quat_index_based()         # Mr, M_deg, M_pauli_lookup, Nr, N_deg, N_pauli_lookup, m, n, Mc, Nc
        self.H_from_index_based()           # H
        self.quat_to_decomp()               # Hx, Hz
        self.decomp_to_bin()                # Hbin, N_rank_check, S_rank, mbin, nbin

        if css_bool:
            self.set_CSS_index_based()      # Mx_CSS, Mz_CSS, Nx_CSS, Nz_CSS, mx
            self.decomp_to_CSS()            # Hx_CSS, Hz_CSS

    def write_CSS_to_alist(self, Hx_CSS, Hz_CSS, file_path = None):
        self.Hx_CSS = Hx_CSS                # Hx_CSS
        self.Hz_CSS = Hz_CSS                # Hz_CSS
        self.CSS_to_decomp()                # Hx, Hz
        self.decomp_to_quat()               # H

        self.decomp_to_bin()                # Hbin, Mbin, Nbin_deg, Nbin, Mbin_deg, mbin, nbin
        self.bin_to_decomp_index_based()    # Mx, Mz, Nx, Nz
        self.set_quat_index_based()         # M, M_deg, M_pauli_lookup, N, N_deg, N_pauli_lookup, m, n

        self.set_CSS_index_based()          # Mx_CSS, Mz_CSS, Nx_CSS, Nz_CSS, mx

        if file_path != None:
            self.write_alist(file_path)  # ALIST

    def write_quat_to_alist(self, H, file_path = None, css_bool = True):
        self.H = H                          # H
        self.quat_to_decomp()               # Hx, Hz

        self.decomp_to_bin()                # Hbin, Mbin, Nbin_deg, Nbin, Mbin_deg, mbin, nbin, N_rank_check
        self.bin_to_decomp_index_based()    # Mx, Mz, Nx, Nz
        self.set_quat_index_based()         # M, M_deg, M_pauli_lookup, N, N_deg, N_pauli_lookup, m, n

        if css_bool:
            self.set_CSS_index_based()      # Mx_CSS, Mz_CSS, Nx_CSS, Nz_CSS, mx
            self.decomp_to_CSS()            # Hx_CSS, Hz_CSS

        if file_path != None:
            self.write_alist(file_path)         # ALIST

    def stabilizer_check(self, error_quat):
        error_bin = np.argwhere(np.hstack((np.isin(error_quat, [1, 3]), np.isin(error_quat, [2, 3])))).flatten()
        if error_bin.size == 0:
            return True
        else:
            return self.S_rank == self.gf2_rank([*self.N_rank_check, sum([2 ** int(entry) for entry in list(error_bin)])])
