import numpy as np

L1 = 2
L2 = 2
N = 5

number = L1 * L2 * 2 - 2


def index(l1, l2, orbit):
    return (l1 * L2 + l2) * 2 + orbit - 1


m = np.zeros([number, number])

for l1 in range(L1):
    for l2 in range(L2):
        if (l1, l2) != (0, 0) and (l1, l2) != (L1 - 1, L2 - 1):
            m[index(l1, l2, 0), index(l1, l2, 1)] = 1
        if l1 != 0:
            m[index(l1 - 1, l2, 1), index(l1, l2, 0)] = 1
        if l2 != 0:
            m[index(l1, l2 - 1, 1), index(l1, l2, 0)] = 1

m += m.T

eigs = np.linalg.eig(m)[0]
eigs = sorted(eigs)

Nu = N // 2
Nd = N - Nu

energy = (sum(eigs[:Nu]) + sum(eigs[:Nd]))/number

print(energy)
