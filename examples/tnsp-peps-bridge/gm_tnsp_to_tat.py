# -*- coding: utf-8 -*-
import os
import sys
import pickle
import TAT
import tetragono as tet
from bridge import bridge

# Read from Honeycomb Hubbard model exported by SJDong's library

help_message = f"usage: {sys.argv[0]} data_file_name"
if len(sys.argv) != 2:
    print(help_message)
    exit(1)
data_file_name = sys.argv[1]
if data_file_name in ["-h", "-help", "--help"]:
    print(help_message)
    exit(0)
with open(data_file_name, "r") as file:
    data = file.read().split("\n")[:-1]
data.reverse()

with open("su.dat", "rb") as file:
    state = pickle.load(file)

state = tet.conversion.simple_update_lattice_to_sampling_lattice(state)

for l1 in range(state.L1):
    for l2 in range(state.L2):
        print(f" reading site {l1},{l2}")
        site_name = f"A{l1+1}_{l2+1}"
        new_tensor = state[l1, l2].copy().zero()
        if (l1, l2) == (0, 0):
            physics_dimension = 4
            new_tensor_merged = new_tensor.merge_edge({"P": ["P1"]})
        elif (l1, l2) == (state.L1 - 1, state.L2 - 1):
            physics_dimension = 4
            new_tensor_merged = new_tensor.merge_edge({"P": ["P0"]})
        else:
            physics_dimension = 16
            new_tensor_merged = new_tensor.merge_edge({"P": ["P1", "P0"]})
        physics_edge = new_tensor_merged.edges("P")

        for d in range(physics_dimension):
            site = bridge(data.pop)
            if site is not None:
                site = site.edge_rename({
                    f"{site_name}.L": "L",
                    f"{site_name}.R": "R",
                    f"{site_name}.U": "U",
                    f"{site_name}.D": "D",
                    f"{site_name}.n": "P",
                    f"TotalN.n": "T"
                })
                symmetry = site.edges("P").segment[0][0]
                shrinker = state.Tensor(["P", "Q"], [[(symmetry, 1)], physics_edge.conjugated()]).zero()
                shrinker[{"Q": d, "P": 0}] = 1
                new_tensor_merged += site.contract(shrinker.conjugate(), {("P", "P")}).edge_rename({"Q": "P"})
        if (l1, l2) == (0, 0):
            state[l1, l2] = new_tensor_merged.split_edge({"P": [("P1", new_tensor.edges("P1").segment)]})
        elif (l1, l2) == (state.L1 - 1, state.L2 - 1):
            state[l1, l2] = new_tensor_merged.split_edge({"P": [("P0", new_tensor.edges("P0").segment)]})
        else:
            state[l1, l2] = new_tensor_merged.split_edge(
                {"P": [
                    ("P1", new_tensor.edges("P1").segment),
                    ("P0", new_tensor.edges("P0").segment),
                ]})
        state[l1, l2] = state[l1, l2].reverse_edge({"L", "R", "U", "D"})

with open("gm.dat", "wb") as file:
    pickle.dump(state, file)
