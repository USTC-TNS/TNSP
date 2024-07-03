#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import TAT
import tetragono as tet
from tetragono.common_tensor.tensor_toolkit import rename_io, kronecker_product


def abstract_state(L1, L2, file_name, T=False, is_complex=False):
    """
    Create a quantum chemistry state from Hamiltonian in openfermion format.
    Every orbit will be put into a site in lattice L1 * L2.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    file_name : str
        A file containing Hamiltonian in openfermion format.
    T : bool
        Whether the total electron number is odd.
    """
    if is_complex:
        Tensor = TAT.FermiZ2.C.Tensor
    else:
        Tensor = TAT.FermiZ2.D.Tensor
    EF = ([(False, 1), (True, 1)], False)
    ET = ([(False, 1), (True, 1)], True)

    CP = Tensor(["O0", "I0", "T"], [EF, ET, ([(True, 1)], False)]).zero_()
    CP[{"O0": (True, 0), "I0": (False, 0), "T": (True, 0)}] = 1
    CM = Tensor(["O0", "I0", "T"], [EF, ET, ([(True, 1)], True)]).zero_()
    CM[{"O0": (False, 0), "I0": (True, 0), "T": (True, 0)}] = 1
    I = Tensor(["O0", "I0"], [EF, ET]).identity_({("I0", "O0")})
    C0daggerC1 = rename_io(CP, [0]).contract(rename_io(CM, [1]), {("T", "T")})
    C0daggerC1daggerC2C3 = kronecker_product(rename_io(C0daggerC1, [1, 2]), rename_io(C0daggerC1, [0, 3]))
    C0daggerC1_conjugated = C0daggerC1.conjugate().edge_rename({"I0": "O0", "I1": "O1", "O0": "I0", "O1": "I1"})

    state = tet.AbstractState(Tensor, L1, L2)
    state.physics_edges[...] = EF
    state.total_symmetry = T

    m = lambda x: (x // L2, x % L2, 0)

    data = np.load(file_name, allow_pickle=True).item()
    single, double = state.attribute["quantum_chemistry_term"] = [[], []]
    for count, [term, coefficient] in enumerate(data.terms.items()):
        tet.show(f"reading {count}/{len(data.terms)}: {term}")
        match term:
            case ((site_0, 1), (site_1, 0)):  # c^ c
                if site_0 != site_1:
                    single.append([
                        (m(site_0), m(site_1)),
                        C0daggerC1 * coefficient,
                        [],
                    ])
                else:
                    # single site N
                    state.hamiltonians[m(site_0), m(site_1)] = C0daggerC1 * coefficient
            case ((site_0, 1), (site_1, 1), (site_2, 0), (site_3, 0)):  # c^ c^ c c
                if len({site_0, site_1, site_2, site_3}) == 4:
                    # c^0 c^1 c2 c3
                    # + c^0 c3 c^1 c2
                    # - c^0 c2 c^1 c3
                    spins = [site_0 % 2, site_1 % 2, site_2 % 2, site_3 % 2]
                    if sum(spins) in [2]:
                        if spins[0] == spins[3]:
                            double.append([
                                (m(site_0), m(site_3)),
                                (+coefficient) * C0daggerC1,
                                (m(site_1), m(site_2)),
                                C0daggerC1_conjugated,
                                [],
                            ])
                        else:
                            double.append([
                                (m(site_0), m(site_2)),
                                (-coefficient) * C0daggerC1,
                                (m(site_1), m(site_3)),
                                C0daggerC1_conjugated,
                                [],
                            ])
                    elif sum(spins) in [0, 4]:
                        double.append([
                            (m(site_0), m(site_3)),
                            (+coefficient / 2) * C0daggerC1,
                            (m(site_1), m(site_2)),
                            C0daggerC1_conjugated,
                            [],
                        ])
                        double.append([
                            (m(site_0), m(site_2)),
                            (-coefficient / 2) * C0daggerC1,
                            (m(site_1), m(site_3)),
                            C0daggerC1_conjugated,
                            [],
                        ])
                    else:
                        tet.showln(f"unrecognized term: {other}")
                elif len({site_0, site_1, site_2, site_3}) == 3:
                    if site_1 == site_2:
                        single.append([(m(site_0), m(site_3)), +coefficient * C0daggerC1, [m(site_1)]])
                    elif site_0 == site_3:
                        single.append([(m(site_1), m(site_2)), +coefficient * C0daggerC1, [m(site_0)]])
                    elif site_0 == site_2:
                        single.append([(m(site_1), m(site_3)), -coefficient * C0daggerC1, [m(site_0)]])
                    elif site_1 == site_3:
                        single.append([(m(site_0), m(site_2)), -coefficient * C0daggerC1, [m(site_1)]])
                    else:
                        tet.showln(f"unrecognized term: {other}")
                else:
                    # double site NN
                    if {site_0, site_1} != {site_2, site_3}:
                        tet.showln(f"unrecognized term: {other}")
                    state.hamiltonians[m(site_0), m(site_1), m(site_2), m(site_3)] = C0daggerC1daggerC2C3 * coefficient
            case ():
                state.hamiltonians[((0, 0, 0),)] = coefficient * I
            case other:
                tet.showln(f"unrecognized term: {other}")
                raise NotImplementedError()
    tet.showln(f"read done, total {len(data.terms)}")
    state.hamiltonians.trace_repeated().sort_points().check_hermite(1e-15)
    tet.showln(f"clean terms, total {len(state.hamiltonians)}")
    return state


def abstract_lattice(L1, L2, D, file_name, T=False, is_complex=False):
    """
    Create a quantum chemistry lattice from Hamiltonian in openfermion format.
    Every orbit will be put into a site in lattice L1 * L2.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The dimension cut for PEPS.
    file_name : str
        A file containing Hamiltonian in openfermion format.
    T : bool
        Whether the total electron number is odd.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, file_name, T=T, is_complex=is_complex))
    D1 = D // 2
    D2 = D - D1
    state.virtual_bond["R"] = [(False, D1), (True, D2)]
    state.virtual_bond["D"] = [(False, D1), (True, D2)]
    return state
