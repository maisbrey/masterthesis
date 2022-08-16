import numpy as np
from numpy.polynomial import Polynomial


def bch_pcm(n, k, g_oct):
    def strip_zeros(a):
        return np.trim_zeros(a, trim='b')

    def gf2_div(dividend, divisor):
        N = len(dividend) - 1
        D = len(divisor) - 1

        if dividend[N] == 0 or divisor[D] == 0:
            dividend, divisor = strip_zeros(dividend), strip_zeros(divisor)

        if not divisor.any():  # if every element is zero
            raise ZeroDivisionError("polynomial division")
        elif D > N:
            q = np.array([])
            return q, dividend

        else:
            u = dividend.astype("uint8")
            v = divisor.astype("uint8")

            m = len(u) - 1
            n = len(v) - 1
            scale = v[n].astype("uint8")
            q = np.zeros((max(m - n + 1, 1),), u.dtype)
            r = u.astype(u.dtype)

            for k in range(0, m - n + 1):
                d = scale and r[m - k].astype("uint8")
                q[-1 - k] = d
                r[m - k - n:m - k + 1] = np.logical_xor(r[m - k - n:m - k + 1], np.logical_and(d, v))

            r = strip_zeros(r)

        return q, r

    g = np.zeros(3 * len(g_oct), dtype=np.uint8)
    for index, entry in enumerate(g_oct):
        g[3 * index:3 * (index + 1)] = np.unpackbits(np.array(entry).astype(np.uint8))[-3:]
    g = g[-(n - k + 1):]

    p = np.zeros(n + 1, dtype=np.uint8)
    p[0] = p[-1] = 1
    h, r = gf2_div(p, g[::-1])  # rightmost coefficient is leading in gf2_div --> we require reciprocal polynome

    H = np.zeros((n - k, n), dtype=np.uint8)

    for shift in np.arange(n - k):
        H[shift, shift:shift + k + 1] = h[::-1]  # h is rightmost, we require h to be leftmost before shifting

    return H


def generate_regular_PCM(m, n, dv):
    dc = int(n / m * dv)
    H_valid = True
    H = np.zeros((m, n), dtype=np.uint8)
    row_count = np.zeros(m, dtype=np.uint8)

    for column in np.arange(n):
        p = (dc - row_count) / np.sum(dc - row_count)
        try:
            row_selection = np.random.choice(np.arange(m)[row_count < dc], dv,
                                             p=p[row_count < dc], replace=False)
        except:
            H_valid = False
            return H, H_valid

        row_count[row_selection] += 1
        H[row_selection, column] = 1

    return H, H_valid


def remove_4cycle(H, m, n):
    success = True
    abort = False
    trials = 0
    while not abort:
        # T contains the number of all shared CNs for all pairs of columns (VNs)
        T = np.dot(H.T, H) - np.diag(np.sum(H, axis=0))
        # v1s and v2s contain
        v1s, v2s = np.argwhere(np.triu(T) >= 2).T
        if v1s.size == 0:
            abort = True
        for v1, v2 in zip(v1s, v2s):
            cncs = np.argwhere(H[:, v1] + H[:, v2] == 2).flatten()
            if cncs.size >= 2:
                cn = cncs[0]
                find_replace = False
                while not find_replace:
                    vni = np.random.choice(np.arange(n)[~np.isin(np.arange(n), [v1, v2])])
                    if np.dot(H[:, v2], H[:, vni]) == 0:
                        cnt = np.random.choice(np.argwhere(H[:, vni]).flatten())
                        temp = H[[cnt, cn], v2]
                        H[[cnt, cn], v2] = H[[cnt, cn], vni]
                        H[[cnt, cn], vni] = temp
                        find_replace = True
        trials += 1
        if trials > 100:
            success = False
            abort = True
    return H, success


def generate_regular_4cycle_free_PCM(m, n, dv):
    success = False
    while not success:
        H, H_valid = generate_regular_PCM(m, n, dv)

        if H_valid:
            H, success = remove_4cycle(H, m, n)
    return H
