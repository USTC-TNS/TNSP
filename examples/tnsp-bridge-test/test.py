from bridge import bridge

with open("./SavingData/NodeData6.dat", "r") as file:
    data = file.read().split("\n")[:-1]
data.reverse()

L1 = 4
L2 = 4
put_sign_in_H = True
# 同一个物理格点，两个自旋有一个merge

pool = {}
for i in range(L1):
    for j in range(L2):
        site_name = f"A{i+1}_{j+1}"
        site = bridge(data.pop).edge_rename({
            f"{site_name}.L": "L",
            f"{site_name}.R": "R",
            f"{site_name}.U": "U",
            f"{site_name}.D": "D",
            f"{site_name}.o1n": "P0",
            f"{site_name}.o2n": "P1",
            f"{site_name}.TotalN": "T"
        })
        pool[i, j, "s"] = site
        rename_env = {"Lambda.D": "D", "Lambda.U": "U", "Lambda.L": "L", "Lambda.R": "R"}
        pool[i, j, "u"] = bridge(data.pop).edge_rename(rename_env)
        pool[i, j, "d"] = bridge(data.pop).edge_rename(rename_env)
        pool[i, j, "r"] = bridge(data.pop).edge_rename(rename_env)
        pool[i, j, "l"] = bridge(data.pop).edge_rename(rename_env)

for i in range(L1):
    for j in range(L2):
        if i != 0:
            diff = pool[i - 1, j, "d"] - pool[i, j, "u"]
            assert diff.norm_max() < 1e-6
        if j != 0:
            diff = pool[i, j - 1, "r"] - pool[i, j, "l"]
            assert diff.norm_max() < 1e-6

import TAT
import tetragono as tet


class FakeEdge:

    def __init__(self, direction):
        self.direction = direction

    def __getitem__(self, x):
        return (list(x), self.direction)


Fedge = FakeEdge(False)
Tedge = FakeEdge(True)


def rename_io(t, m):
    res = {}
    for i, j in m.items():
        res[f"I{i}"] = f"I{j}"
        res[f"O{i}"] = f"O{j}"
    return t.edge_rename(res)


def dot(res, *b):
    for i in b:
        res = res.contract(i, set())
    return res


# Ops Before Merge

# EPR pair: (F T)
CP = TAT.Fermi.D.Tensor(["O", "I", "T"], [Fedge[0, 1], Tedge[0, -1], Fedge[-1,]]).range(1)
CM = TAT.Fermi.D.Tensor(["O", "I", "T"], [Fedge[0, 1], Tedge[0, -1], Tedge[+1,]]).range(1)
C0C1 = rename_io(CP, {"": 0}).contract(rename_io(CM, {"": 1}), {("T", "T")})
C1C0 = rename_io(CP, {"": 1}).contract(rename_io(CM, {"": 0}), {("T", "T")})
CC = C0C1 + C1C0  # rank = 4
# print(CC)

I = TAT.Fermi.D.Tensor(["O", "I"], [Fedge[0, 1], Tedge[0, -1]]).identity({("I", "O")})

N = TAT.Fermi.D.Tensor(["O", "I"], [Fedge[0, 1], Tedge[0, -1]]).zero()
N[{"I": 1, "O": 1}] = 1

# Ops After Merge

# site1 up: 0
# site2 up: 1
# site1 down: 2
# site2 down: 3
# CSCS = CC(0,1)I(2)I(3) + I(0)I(1)CC(2,3)
CSCS = dot(
    rename_io(CC, {
        0: 0,
        1: 1
    }),
    rename_io(I, {"": 2}),
    rename_io(I, {"": 3}),
) + dot(
    rename_io(CC, {
        0: 2,
        1: 3
    }),
    rename_io(I, {"": 0}),
    rename_io(I, {"": 1}),
)
CSCS = CSCS.merge_edge({
    "I0": ["I0", "I2"],
    "O0": ["O0", "O2"],
    "I1": ["I1", "I3"],
    "O1": ["O1", "O3"],
}, put_sign_in_H, {"O0", "O1"})
# print("CC", CSCS)
# print("CC*", CSCS.conjugate().edge_rename({"I0": "O0", "O0": "I0", "I1": "O1", "O1": "I1"}))
# print("CC*-CC", CSCS.conjugate().edge_rename({"I0": "O0", "O0": "I0", "I1": "O1", "O1": "I1"}) - CSCS)
# print()

NN = dot(rename_io(N, {"": 0}), rename_io(N, {"": 1}))
NN = NN.merge_edge({
    "I0": ["I0", "I1"],
    "O0": ["O0", "O1"],
}, put_sign_in_H, {"O0"})
# print("NN", NN)
# print("NN*", NN.conjugate().edge_rename({"I0": "O0", "O0": "I0"}))
# print("NN*-NN", NN.conjugate().edge_rename({"I0": "O0", "O0": "I0"}) - NN)
# exit()

state = tet.AbstractState(TAT.Fermi.D.Tensor, L1, L2)
# state.total_symmetry = 28
t = -1
U = 12
for i in range(L1):
    for j in range(L2):
        if (i, j) != (0, 0):
            state.physics_edges[i, j, 0] = pool[i, j, "s"].edges("P0")
            state.hamiltonians[(i, j, 0),] = U * NN
        if (i, j) != (L1 - 1, L2 - 1):
            state.physics_edges[i, j, 1] = pool[i, j, "s"].edges("P1")
            state.hamiltonians[(i, j, 1),] = U * NN
        if (i, j) != (0, 0) and (i, j) != (L1 - 1, L2 - 1):
            state.hamiltonians[(i, j, 0), (i, j, 1)] = t * CSCS
        if i != 0:
            state.hamiltonians[(i - 1, j, 1), (i, j, 0)] = t * CSCS
        if j != 0:
            state.hamiltonians[(i, j - 1, 1), (i, j, 0)] = t * CSCS

state = tet.AbstractLattice(state)
for i in range(L1):
    for j in range(L2):
        if i != 0:
            state.virtual_bond[i, j, "U"] = pool[i, j, "s"].edges("U")
        if j != 0:
            state.virtual_bond[i, j, "L"] = pool[i, j, "s"].edges("L")

state = tet.SimpleUpdateLattice(state)
for i in range(L1):
    for j in range(L2):
        # Need to use tnsp tensor, because our T position is different
        state[i, j] = pool[i, j, "s"]
        if i != 0:
            state.environment[i, j, "U"] = pool[i, j, "u"]
        if j != 0:
            state.environment[i, j, "L"] = pool[i, j, "l"]

for i in range(L1):
    for j in range(L2):
        # SJ.Dong contract env into state
        state[i, j] = state._try_multiple(state[i, j], i, j, "L", True)
        state[i, j] = state._try_multiple(state[i, j], i, j, "R", True)
        state[i, j] = state._try_multiple(state[i, j], i, j, "U", True)
        state[i, j] = state._try_multiple(state[i, j], i, j, "D", True)

import pickle
prefix = "SJ8"
for i in range(10):
    with open(prefix+f"ME{i}", "wb") as file:
        pickle.dump(state, file)
        state.update(1, 0.01, 6)

#state.initialize_auxiliaries(-1)
#print(state.observe_energy())
#state.initialize_auxiliaries(-1)
#print(state.observe_energy())
