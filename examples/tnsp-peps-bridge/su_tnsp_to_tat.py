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
for l1 in range(L1):
    for l2 in range(L2):
        print(f" reading site {l1},{l2}")
        print(f"  reading site tensor")
        site_name = f"A{l1+1}_{l2+1}"
        site = bridge(data.pop).edge_rename({
            f"{site_name}.L": "L",
            f"{site_name}.R": "R",
            f"{site_name}.U": "U",
            f"{site_name}.D": "D",
            f"{site_name}.o1n": "P0",
            f"{site_name}.o2n": "P1",
            f"{site_name}.TotalN": "T"
        })
        pool[l1, l2, "s"] = site
        print(f"  reading env tensor u")
        pool[l1, l2, "u"] = try_rename_env(bridge(data.pop))
        print(f"  reading env tensor d")
        pool[l1, l2, "d"] = try_rename_env(bridge(data.pop))
        print(f"  reading env tensor r")
        pool[l1, l2, "r"] = try_rename_env(bridge(data.pop))
        print(f"  reading env tensor l")
        pool[l1, l2, "l"] = try_rename_env(bridge(data.pop))

for l1 in range(L1):
    for l2 in range(L2):
        if l1 != 0:
            diff = pool[l1 - 1, l2, "d"] - pool[l1, l2, "u"]
            assert diff.norm_max() < 1e-6
        if l2 != 0:
            diff = pool[l1, l2 - 1, "r"] - pool[l1, l2, "l"]
            assert diff.norm_max() < 1e-6

CSCS = tet.common_tensor.Fermi_Hubbard.CSCS.to(float)
NN = tet.common_tensor.Fermi_Hubbard.NN.to(float)

state = tet.AbstractState(TAT.Fermi.D.Tensor, L1, L2)
for l1 in range(L1):
    for l2 in range(L2):
        if (l1, l2) != (0, 0):
            state.physics_edges[l1, l2, 0] = pool[l1, l2, "s"].edges("P0")
            state.hamiltonians[(l1, l2, 0),] = U * NN
        if (l1, l2) != (L1 - 1, L2 - 1):
            state.physics_edges[l1, l2, 1] = pool[l1, l2, "s"].edges("P1")
            state.hamiltonians[(l1, l2, 1),] = U * NN
        if (l1, l2) != (0, 0) and (l1, l2) != (L1 - 1, L2 - 1):
            state.hamiltonians[(l1, l2, 0), (l1, l2, 1)] = t * CSCS
        if l1 != 0:
            state.hamiltonians[(l1 - 1, l2, 1), (l1, l2, 0)] = t * CSCS
        if l2 != 0:
            state.hamiltonians[(l1, l2 - 1, 1), (l1, l2, 0)] = t * CSCS

state = tet.AbstractLattice(state)
for l1 in range(L1):
    for l2 in range(L2):
        if l1 != 0:
            state.virtual_bond[l1, l2, "U"] = pool[l1, l2, "s"].edges("U")
        if l2 != 0:
            state.virtual_bond[l1, l2, "L"] = pool[l1, l2, "s"].edges("L")

state = tet.SimpleUpdateLattice(state)
for l1 in range(L1):
    for l2 in range(L2):
        # Need to use tnsp tensor, because our T position is different
        state[l1, l2] = pool[l1, l2, "s"]
        if l1 != 0:
            state.environment[l1, l2, "U"] = pool[l1, l2, "u"]
        if l2 != 0:
            state.environment[l1, l2, "L"] = pool[l1, l2, "l"]

with open("su.dat", "wb") as file:
    pickle.dump(state, file)
