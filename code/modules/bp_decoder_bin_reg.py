import numpy as np

class DecoderBinaryBase:
    lookup_xp = np.linspace(0, 8, 16)
    lookup_fp = np.log(1 + np.exp(-lookup_xp))

    def __init__(self, Mr, Mc, Nr, Nc, epsilon):

        self.Mr, self.Mc = Mr, Mc
        self.Nr, self.Nc = Nr, Nc
        self.epsilon = epsilon

        self.M_deg = Nr.shape[1]
        self.N_deg = Mr.shape[1]

        self.extrinsic_LU = np.array([np.r_[0:CN, CN + 1:self.M_deg] for CN in np.arange(self.M_deg)])

        self.m, self.n = Nr.shape[0], Mr.shape[0]

        self.Lambda = np.log((1.5 - self.epsilon) / self.epsilon) * np.ones(self.n)
        self.Gamma = self.Lambda.copy()

        self.CN_to_VN_data = (-1) * np.ones(self.Nr.shape)
        self.VN_to_CN_data = (-1) * np.ones(self.Mr.shape)

        self.forward = np.zeros(self.Nr.shape)
        self.backward = np.zeros(self.Nr.shape)

        self.error = np.zeros(self.n, dtype=np.uint8)
        self.z = np.zeros(self.m, dtype=np.uint8)

        self.error_estimate = np.zeros(self.n, dtype=np.uint8)
        self.z_estimate = np.zeros(self.m, dtype=np.uint8)

        self.Gamma_history_list = [self.Gamma.copy()]

        self.normalize_CN_msgs = False
        self.alpha_c = 1
        self.a = 1
        self.b = 0
        self.iter_count = 0
        self.schedule = []

    def set_error(self, error):
        self.error = error
        self.z = self.compute_syndrome(self.error)

    def compute_syndrome(self, error):
        error_indices = np.argwhere(error).flatten()
        z = np.zeros(self.m, dtype=np.uint16)
        for VN in error_indices:
            z[self.Mr[VN][self.Mr[VN] != -1]] += 1
        return (z % 2).astype(np.uint8)

    def update_estimates(self):
        self.error_estimate = (self.Gamma < 0).astype(np.uint8)
        self.z_estimate = self.compute_syndrome(self.error_estimate)

    def log_jacobian(self, x, y):
        return np.max(np.array([x, y]), axis=0) + np.interp(np.abs(x - y), self.lookup_xp, self.lookup_fp, right=0)

    def box_plus(self, x, y):
        return self.log_jacobian(np.zeros_like(x), x + y) - self.log_jacobian(x, y)

    def initialization(self):
        self.VN_to_CN_data[:] = self.Gamma.reshape(-1, 1)

    def early_stopping(self):
        self.update_estimates()
        return np.all(self.z == self.z_estimate)

    def update_CN_subset(self, m_):
        messages = self.VN_to_CN_data[self.Nr[m_], self.Nc[m_]]
        self.forward[:, 0] = messages[:, 0]
        self.backward[:, -1] = messages[:, -1]

        for index in np.arange(1, self.M_deg - 1):
            self.forward[:, index] = self.box_plus(messages[:, index], self.forward[:, index - 1])
            self.backward[:, -(index + 1)] = self.box_plus(messages[:, -(index + 1)], self.backward[:, -index])

        self.CN_to_VN_data[m_] = (-1) ** self.z[m_].reshape(-1, 1) * \
                                 np.hstack((self.backward[:, 1].reshape(-1, 1),
                                            self.box_plus(self.forward[:, :-2], self.backward[:, 2:]),
                                            self.forward[:, -2].reshape(-1, 1)))

        if self.normalize_CN_msgs:
            self.CN_to_VN_data[m_] *= self.alpha_c

    def update_VN_subset(self, n_):
        messages = self.CN_to_VN_data[self.Mr[n_], self.Mc[n_]]
        messages[self.Mr[n_] == -1] = 0
        self.Gamma[n_] = self.Lambda[n_] + np.sum(messages, axis=1)
        self.VN_to_CN_data[n_] = self.Gamma[n_].reshape(-1, 1) - messages
        self.VN_to_CN_data[n_][self.Mr[n_] == -1] = 0

    def decode(self, error, update, N_iter, init=True, early_stopping=True, log_history=False):
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


class DecoderBinaryParallel(DecoderBinaryBase):
    def update_CNs(self):
        self.update_CN_subset(np.arange(self.m))

    def update_VNs(self):
        self.update_VN_subset(np.arange(self.n))

    def parallel(self):
        if self.normalize_CN_msgs:
            self.alpha_c = 1 - (1 - self.a) * np.float_power(2, -self.b * self.iter_count)
        self.update_CNs()
        self.update_VNs()
        self.iter_count += 1
        if self.early_stopping():
            return True


class DecoderBinarySerial(DecoderBinaryBase):

    def update_CN_subset(self, CNs, ports):
        messages = self.VN_to_CN_data[self.Nr[CNs], self.Nc[CNs]]
        batch_size = messages.shape[0]
        range_array = np.arange(batch_size)

        batch = messages[range_array.reshape(-1, 1), self.extrinsic_LU[ports]].T

        updates = (-1) ** self.z[CNs] * self.box_plus(self.box_plus(batch[0], batch[1]), batch[2])

        if self.normalize_CN_msgs:
            updates *= self.alpha_c

        self.CN_to_VN_data[CNs, ports] = updates

    def update_VN_subset(self, n_):
        CNs, ports = self.Mr[n_], self.Mc[n_]
        self.update_CN_subset(CNs[CNs != -1], ports[CNs != -1])
        messages = self.CN_to_VN_data[CNs, ports]
        messages[CNs == -1] = 0
        self.Gamma[n_] = self.Lambda[n_] + np.sum(messages, axis=1)
        self.VN_to_CN_data[n_] = self.Gamma[n_].reshape(-1, 1) - messages
        self.VN_to_CN_data[n_][self.Mr[n_] == -1] = 0

    def serial_VN_schedule(self):
        if self.normalize_CN_msgs:
            self.alpha_c = 1 - (1 - self.a) * np.float_power(2, -self.b * self.iter_count)
        for n_ in self.schedule:
            self.update_VN_subset(n_)
        self.iter_count += 1
        if self.early_stopping():
            return True