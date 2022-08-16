import numpy as np
import galois


class DecoderBase:
    GF4 = galois.GF(4)
    lookup_xp = np.linspace(0, 8, 16)
    lookup_fp = np.log(1 + np.exp(-lookup_xp))

    def __init__(self, Mr, Mc, M_pauli_lookup, Nr, Nc, N_pauli_lookup, epsilon):

        self.Mr, self.Mc, self.M_pauli_lookup = Mr, Mc, M_pauli_lookup
        self.Nr, self.Nc, self.N_pauli_lookup = Nr, Nc, N_pauli_lookup
        self.epsilon = epsilon

        self.M_deg = Nr.shape[1]
        self.N_deg = Mr.shape[1]

        self.extrinsic_LU = np.array([np.r_[0:CN, CN + 1:self.M_deg] for CN in np.arange(self.M_deg)])

        self.M_pauli_mask = self.M_pauli_lookup == 1

        self.m, self.n = Nr.shape[0], Mr.shape[0]

        self.Lambda = np.log(3 * (1 - self.epsilon) / self.epsilon) * np.ones((self.n, 3))
        self.Gamma = self.Lambda.copy()

        self.CN_to_VN_data = (-1) * np.ones(self.Nr.shape)
        self.VN_to_CN_data = (-1) * np.ones(self.Mr.shape)

        self.forward = np.zeros(self.Nr.shape)
        self.backward = np.zeros(self.Nr.shape)

        self.error = self.GF4(np.zeros(self.n, dtype=np.uint8))
        self.z = np.zeros(self.m, dtype=np.uint8)

        self.error_estimate = self.GF4(np.zeros(self.n, dtype=np.uint8))
        self.z_estimate = np.zeros(self.m, dtype=np.uint8)

        self.Gamma_history_list = [self.Gamma.copy()]

        self.normalize_CN_msgs = False
        self.alpha_c = 1
        self.a = 1
        self.b = 0
        self.iter_count = 0
        self.schedule = []

    def set_error(self, error):
        self.error = self.GF4(error)
        self.z = self.compute_syndrome(self.error)

    def compute_syndrome(self, error):
        error_indices = np.argwhere(error).flatten()
        z = self.GF4(np.zeros(self.m, dtype=np.uint16))
        for VN in error_indices:
            z[self.Mr[VN][self.Mr[VN] != -1]] += error[VN] ** 2 * self.GF4(self.M_pauli_lookup[VN][self.Mr[VN] != -1])
        return (z > 1).astype(np.uint8)

    def update_estimates(self):
        self.error_estimate = self.GF4(np.argmin(np.hstack((np.zeros((self.Gamma.shape[0], 1)), self.Gamma)), axis=1))
        self.z_estimate = self.compute_syndrome(self.error_estimate)

    def log_jacobian(self, x, y):
        return np.max(np.array([x, y]), axis=0) + np.interp(np.abs(x - y), self.lookup_xp, self.lookup_fp, right=0)

    def box_plus(self, x, y):
        return self.log_jacobian(np.zeros_like(x), x + y) - self.log_jacobian(x, y)

    def get_lambda_lookup(self, Gamma, n):
        lambda_lookup = np.zeros((n, 2))

        for eta in [1, 2]:
            lambda_lookup[:, eta - 1] = self.log_jacobian(np.zeros(n), -Gamma[:, eta - 1]) - \
                                        self.log_jacobian(*(-Gamma[:, np.r_[0:eta - 1, eta:3]].T))
        return lambda_lookup

    def initialization(self):
        self.VN_to_CN_data = self.get_lambda_lookup(self.Gamma, self.n)[np.arange(self.n).reshape(1, -1),
                                                                        self.M_pauli_lookup.T - 1].T

    def early_stopping(self):
        self.update_estimates()
        return np.all(self.z == self.z_estimate)

    def update_CNs(self):
        messages = self.VN_to_CN_data[self.Nr, self.Nc]
        self.forward[:, 0] = messages[:, 0]
        self.backward[:, -1] = messages[:, -1]
        for index in np.arange(1, self.Mr.shape[1] - 1):
            self.forward[:, index] = self.box_plus(messages[:, index], self.forward[:, index - 1])
            self.backward[:, -(index + 1)] = self.box_plus(messages[:, -(index + 1)], self.backward[:, -index])

        self.CN_to_VN_data = (-1) ** np.tile(self.z, (self.Nr.shape[1], 1)).T * \
                             np.hstack((self.backward[:, 1].reshape(-1, 1),
                                        self.box_plus(self.forward[:, :-2], self.backward[:, 2:]),
                                        self.forward[:, -2].reshape(-1, 1)))
        if self.normalize_CN_msgs:
            self.CN_to_VN_data *= self.alpha_c

    def update_VNs(self):
        messages = self.CN_to_VN_data[self.Mr, self.Mc]
        messages[self.Mr == -1] = 0

        M1 = np.sum(np.where(~self.M_pauli_mask, messages, 0), axis=1)
        M2 = np.sum(np.where(self.M_pauli_mask, messages, 0), axis=1)

        self.Gamma = self.Lambda + np.array([M1, M2, M1 + M2]).T
        self.VN_to_CN_data = self.get_lambda_lookup(self.Gamma, self.n)[np.arange(self.n).reshape(1, -1),
                                                                        self.M_pauli_lookup.T - 1].T - messages
        self.VN_to_CN_data[self.Mr == -1] = 0


    def decode(self, error, update, N_iter=12, init=True, early_stopping=True, log_history=False):
        self.set_error(error)
        self.update_estimates()
        if init:
            self.initialization()

        for _ in range(N_iter):
            if log_history:
                self.Gamma_history_list.append(self.Gamma.copy())
            if update() and early_stopping:
                break

        if log_history:
            self.Gamma_history_list.append(self.Gamma.copy())

        return self.error_estimate


class DecoderParallel(DecoderBase):
    def parallel(self):
        if self.normalize_CN_msgs:
            self.alpha_c = 1 - (1 - self.a) * np.float_power(2, -self.b * self.iter_count)
        self.update_CNs()
        self.update_VNs()
        self.iter_count += 1
        if self.early_stopping():
            return True


class DecoderSerial(DecoderBase):

    def update_CN_subset(self, CNs, ports):
        messages = self.VN_to_CN_data[self.Nr[CNs], self.Nc[CNs]]
        batch_size = messages.shape[0]
        range_array = np.arange(batch_size)

        batch = messages[range_array.reshape(-1, 1), self.extrinsic_LU[ports]].T

        updates = (-1) ** self.z[CNs] * self.box_plus(self.box_plus(self.box_plus(batch[0], batch[1]),
                                                                    self.box_plus(batch[2], batch[3])),
                                                      self.box_plus(batch[4], batch[5]))

        if self.normalize_CN_msgs:
            updates *= self.alpha_c

        self.CN_to_VN_data[CNs, ports] = updates

    def update_VN_subset(self, n_):
        CNs, ports = self.Mr[n_], self.Mc[n_]
        self.update_CN_subset(CNs[CNs != -1], ports[CNs != -1])
        messages = self.CN_to_VN_data[CNs, ports]
        messages[CNs == -1] = 0

        M1 = np.sum(np.where(~self.M_pauli_mask[n_], messages, 0), axis=1)
        M2 = np.sum(np.where(self.M_pauli_mask[n_], messages, 0), axis=1)

        self.Gamma[n_] = self.Lambda[n_] + np.array([M1, M2, M1 + M2]).T
        self.VN_to_CN_data[n_] = self.get_lambda_lookup(self.Gamma[n_], n_.size)[np.arange(n_.size).reshape(1, -1),
                                                                                 self.M_pauli_lookup[
                                                                                     n_].T - 1].T - messages
        self.VN_to_CN_data[n_][self.Mr[n_] == -1] = 0

    def serial_VN(self):
        if self.normalize_CN_msgs:
            self.alpha_c = 1 - (1 - self.a) * np.float_power(2, -self.b * self.iter_count)
        for n_ in np.arange(self.n):
            self.update_VN_subset(np.array([n_]))
        self.iter_count += 1
        if self.early_stopping():
            return True

    def serial_VN_schedule(self):
        if self.normalize_CN_msgs:
            self.alpha_c = 1 - (1 - self.a) * np.float_power(2, -self.b * self.iter_count)
        for n_ in self.schedule:
            self.update_VN_subset(n_)
        self.iter_count += 1
        if self.early_stopping():
            return True