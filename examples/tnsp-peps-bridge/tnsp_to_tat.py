import os
import sys
import pickle
import TAT
import tetragono as tet
from bridge import bridge

# Read from Honeycomb Hubbard model exported by SJDong's library

help_message = f"usage: {sys.argv[0]} info_file_name"
if len(sys.argv) != 2:
    print(help_message)
    exit(1)
info_file_name = sys.argv[1]
if info_file_name in ["-h", "-help", "--help"]:
    print(help_message)
    exit(0)
with open(info_file_name, "r") as file:
    info_data = [line.strip() for line in file.read().split("\n") if line.strip() != ""]
directory_name = os.path.dirname(info_file_name)
data_file_name = os.path.join(directory_name, info_data[-1])
with open(data_file_name, "r") as file:
    data = file.read().split("\n")[:-1]
data.reverse()

L1 = [int(line.split()[1]) for line in info_data if line.split()[0] == "L1"][0]
L2 = [int(line.split()[1]) for line in info_data if line.split()[0] == "L2"][0]
t = [float(line.split()[1]) for line in info_data if line.split()[0] == "t"][0]
U = [float(line.split()[1]) for line in info_data if line.split()[0] == "U"][0]
print(f" lattice {L1}x{L2}, t={t}, U={U}")


def try_rename_env(tensor):
    if tensor != None:
        return tensor.edge_rename({"Lambda.D": "D", "Lambda.U": "U", "Lambda.L": "L", "Lambda.R": "R"})


pool = {}
for i in range(L1):
    for j in range(L2):
        print(f" reading site {i},{j}")
        print(f"  reading site tensor")
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
        print(f"  reading env tensor u")
        pool[i, j, "u"] = try_rename_env(bridge(data.pop))
        print(f"  reading env tensor d")
        pool[i, j, "d"] = try_rename_env(bridge(data.pop))
        print(f"  reading env tensor r")
        pool[i, j, "r"] = try_rename_env(bridge(data.pop))
        print(f"  reading env tensor l")
        pool[i, j, "l"] = try_rename_env(bridge(data.pop))

for i in range(L1):
    for j in range(L2):
        if i != 0:
            diff = pool[i - 1, j, "d"] - pool[i, j, "u"]
            assert diff.norm_max() < 1e-6
        if j != 0:
            diff = pool[i, j - 1, "r"] - pool[i, j, "l"]
            assert diff.norm_max() < 1e-6

CSCS = tet.common_variable.Fermi_Hubbard.CSCS.to(float)
NN = tet.common_variable.Fermi_Hubbard.NN.to(float)

state = tet.AbstractState(TAT.Fermi.D.Tensor, L1, L2)
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

with open("state.dat", "wb") as file:
    pickle.dump(state, file)
