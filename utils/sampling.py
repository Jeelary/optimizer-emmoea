import numpy as np
from math import comb
from itertools import combinations


def UniformPoint(N, M, method='NBI'):
    if method == 'NBI':
        return NBI(N, M)
    elif method == 'Latin':
        return Latin(N, M)

def nChoosek(n, k):
    if k > n:
        return 0
    if k * 2 > n:
        k = n - k
    if k == 0:
        return 1

    result = n
    for i in range(2, k + 1):
        result *= n - i + 1
        result //= i

    return result

def getSLRP(num_ind, num_obj):
    H1 = 1
    while nChoosek(H1 + num_obj, num_obj - 1) <= num_ind:
        H1 += 1
    npoints = nChoosek(H1 + num_obj - 1, num_obj - 1)
    points = np.zeros((npoints, num_obj))
    points2 = np.zeros((npoints, num_obj))

    N = H1 + num_obj - 1
    M = num_obj - 1

    istart = np.zeros(N)
    iend = np.zeros(N)
    imass = np.zeros(N)

    for i in range(M):
        istart[i] = i + 1
        iend[i] = N - M + i + 2
        imass[i] = i + 1

    counter = 0
    while True:
        imass[M - 1] = istart[M - 1]
        while imass[M - 1] != iend[M - 1]:
            for j in range(M):
                points[counter][j] = imass[j]

            imass[M - 1] += 1
            counter += 1

        NFinished = 0
        for i in range(M - 1, -1, -1):
            if imass[i] == iend[i]:
                NFinished += 1
            if imass[i] == iend[i]:
                if i > 0:
                    imass[i - 1] += 1
                    for j in range(i, M):
                        istart[j] = imass[i - 1] + j - i + 1
                        imass[j] = istart[j]

        if NFinished == M:
            break

    for i in range(npoints):
        for j in range(M):
            points[i][j] = points[i][j] - j - 1
            points2[i][j + 1] = points[i][j]
        points2[i][0] = 0

    for i in range(npoints):
        points[i][M] = points[npoints - 1][0]
        for j in range(M + 1):
            points[i][j] = (points[i][j] - points2[i][j]) / H1
            if points[i][j] < 1E-6:
                points[i][j] = 1E-6

    return points

def NBI(N, M):
    H1 = 1
    # print('N:', N)
    while comb(H1 + M - 1, M - 1) <= N:
        H1 += 1
    H1 -= 1
    # print("H1:", H1)

    W = SimplexLattice(H1, M)
    # print('SL:', np.std(W))

    if H1 < M:
        H2 = 0
        while comb(H1 + M - 1, M - 1) + comb(H2 + M - 1, M - 1) <= N:
            H2 += 1
        H2 -= 1

        if H2 > 0:
            W2 = SimplexLattice(H2, M)
            W2 = W2 * 0.5 + 1.0 / (2 * M)
            W = np.vstack((W, W2))

    W = np.maximum(W, 1e-6)
    return W

def SimplexLattice(H: int, M: int) -> np.ndarray:
    if M == 1:
        return np.ones((1, 1))

    W = []
    for c in combinations(range(H + M - 1), M - 1):
        w = np.zeros(M)
        prev = -1
        for i, val in enumerate(c):
            w[i] = val - prev - 1
            prev = val
        w[M - 1] = H + M - 1 - prev - 1
        W.append(w)

    W = np.array(W) / H
    return W

def Latin(N: int, M: int) -> np.ndarray:
    W = np.random.rand(N, M)
    ranks = np.argsort(W, axis=0)
    W = (ranks + np.random.rand(N, M)) / N
    return W
