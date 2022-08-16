import numpy as np
import galois
import itertools

def generate_weight_1_errors_index_based(n):
    error_indices = np.zeros((3 * n, 2), dtype = np.uint8)
    index = 0
    for error_index in np.arange(n):
        for pauli_error in np.arange(1, 4):
            error_indices[index] = error_index, pauli_error
            index += 1
    return error_indices

def generate_weight_2_errors_index_based(n):
    error_indices = np.zeros((9 * int(n * (n - 1) / 2), 4), dtype = np.uint8)

    index = 0
    for error_1_index in np.arange(n):
        for error_2_index in np.arange(error_1_index + 1, n):
            for error_1, error_2 in itertools.product(np.arange(1, 4), np.arange(1, 4)):
                error_indices[index] = error_1_index, error_2_index, error_1, error_2
                index += 1
    return error_indices

def generate_N_weight_w_errors(n, w, N_weight):
    errors = np.zeros((N_weight, n), dtype=np.uint8)
    for error in errors:
        error[np.random.choice(n, w, replace=False)] = np.random.choice(np.arange(1, 4), w, replace=True)
    return errors

def generate_N_errors_length_n_epsilon(N_errors, n, epsilon):
    return np.random.choice(4, (N_errors, n), p = [1-epsilon, epsilon/3, epsilon/3, epsilon/3]).astype(np.uint8)

def compute_syndrome_index_based(M, N_deg, M_pauli_lookup, m, error_index):
    GF4 = galois.GF(4)
    syndrome = GF4(np.zeros(m, dtype = np.uint8))
    for VN, pauli_error in zip(error_index[:int(error_index.size/2)], error_index[int(error_index.size/2):]):
        syndrome[M[VN][:N_deg[VN]]] += GF4(pauli_error) ** 2 * GF4(M_pauli_lookup[VN][:N_deg[VN]])
    return np.argwhere((syndrome > 1)).flatten()

def compute_syndrome(H, error):
    GF4 = galois.GF(4)
    return (np.dot(GF4(H), GF4(error) ** 2) > 1).astype(np.uint8)

def p_from_Gamma(Gamma):
    exp_Gamma = np.exp(-Gamma)
    return np.hstack((np.ones((exp_Gamma.shape[0], 1)), exp_Gamma)) / (1 + np.sum(exp_Gamma, axis = 1)[:, np.newaxis])

def p_from_Gamma_array(Gamma_array):
    exp_Gamma = np.exp(-Gamma_array)
    return np.concatenate((np.ones((exp_Gamma.shape[0], exp_Gamma.shape[1], 1)), exp_Gamma), axis = -1) / (1 + np.sum(exp_Gamma, axis = -1)[:, :, np.newaxis])

def p_from_Gamma_array_bin(Gamma_array):
    exp_Gamma = np.exp(-Gamma_array)
    p1 = (1 / (1 + exp_Gamma))[:, :, np.newaxis]
    return np.concatenate((1-p1, p1), axis = -1)

def output_schedule(n1, m1, n2, m2):
    schedule = []
    for block in np.arange(n1 * n2).reshape(n1, n2):
        schedule.append(block)
    for block in (n1 * n2 + np.arange(m1 * m2)).reshape(m1, m2):
        schedule.append(block)
    return schedule

def construct_schedule(Mr):
    VN = 0
    CN_stack = list(Mr[VN][Mr[VN] != -1])
    schedule = [[0]]
    while VN < Mr.shape[0] - 1:
        VN += 1
        CNs = list(Mr[VN][Mr[VN] != -1])
        if np.any(np.isin(CNs, CN_stack)):
            schedule.append([VN])
            CN_stack = list(CNs)
        else:
            CN_stack.extend(CNs)
            schedule[-1].append(VN)
    for idx, entry in enumerate(schedule):
        schedule[idx] = np.array(entry)
    return schedule

def matr_chessboard(A, B, C, D, top_left = 1):
    matr = np.tile(np.repeat(np.repeat(np.eye(2, dtype = np.uint8), B, axis = 1), A, axis = 0), (C, D))
    if top_left:
        return matr[:A*C, :B*D]
    else:
        return matr[A:A*(C+1), :B*D]