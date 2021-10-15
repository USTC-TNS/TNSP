import TAT

fermi_system = True


class Square:
    if fermi_system:
        Tensor = TAT.Fermi.D.Tensor

        class FakeEdge:

            def __init__(self, direction):
                self.direction = direction

            def __getitem__(self, x):
                return (list(x), self.direction)

        Fedge = FakeEdge(False)
        Tedge = FakeEdge(True)
        hamiltonian = Tensor(["O1", "O0", "I0", "I1"], [Fedge[(0, 1), (1, 1)], Fedge[(0, 1), (1, 1)], Tedge[(0, 1), (-1, 1)], Tedge[(0, 1), (-1, 1)]]).zero()
        hamiltonian[{"O0": (1, 0), "O1": (0, 0), "I0": (0, 0), "I1": (-1, 0)}] = 1
        hamiltonian[{"O0": (0, 0), "O1": (1, 0), "I0": (-1, 0), "I1": (0, 0)}] = 1
    else:
        Tensor = TAT.No.D.Tensor
        Fedge = Tedge = None
        hamiltonian = Tensor(["O0", "O1", "I0", "I1"], [2, 2, 2, 2]).zero()
        __hamiltonian_block = hamiltonian.blocks[hamiltonian.names]
        __hamiltonian_block[0, 0, 0, 0] = 1 / 4.
        __hamiltonian_block[0, 1, 0, 1] = -1 / 4.
        __hamiltonian_block[1, 0, 1, 0] = -1 / 4.
        __hamiltonian_block[1, 1, 1, 1] = 1 / 4.
        __hamiltonian_block[0, 1, 1, 0] = 2 / 4.
        __hamiltonian_block[1, 0, 0, 1] = 2 / 4.

    print("System Hamiltonian is", hamiltonian)

    def __init__(self, N, M, P, D):
        self.N = N
        self.M = M
        self.P = P
        self.D = D
        print(f"System size is {N}*{M}")
        print(f"Bond dimension is {D}")
        print(f"Particle number is {P} (it is unused if it is not fermi system)")

        print("Create lattice")
        Tensor = self.Tensor
        Tedge = self.Tedge
        Fedge = self.Fedge
        network = [[None for m in range(M)] for n in range(N)]
        particle_each_row = P / N
        for n in range(N):
            for m in range(M):
                names = []
                edges = []
                if fermi_system:
                    if n == m == 0:
                        names.append("T")
                        edges.append(Tedge[(-P, 1),])
                if n != 0:
                    names.append(f"U{n}_{m}")
                    if fermi_system:
                        if m != 0:
                            edges.append(Tedge[(0, D),])
                        else:
                            Q = int(P * (N - n) / N)
                            edges.append(Tedge[(-Q + 1, D), (-Q, D), (-Q - 1, D)])
                    else:
                        edges.append(D)
                if n != N - 1:
                    names.append(f"D{n}_{m}")
                    if fermi_system:
                        if m != 0:
                            edges.append(Fedge[(0, D),])
                        else:
                            Q = int(P * (N - n - 1) / N)
                            edges.append(Fedge[(Q - 1, D), (Q, D), (Q + 1, D)])
                    else:
                        edges.append(D)
                if m != 0:
                    names.append(f"L{n}_{m}")
                    if fermi_system:
                        Q = int(particle_each_row * (M - m) / M)
                        edges.append(Tedge[(-Q + 1, D), (-Q, D), (-Q - 1, D)])
                    else:
                        edges.append(D)
                if m != M - 1:
                    names.append(f"R{n}_{m}")
                    if fermi_system:
                        Q = int(particle_each_row * (M - m - 1) / M)
                        edges.append(Fedge[(Q - 1, D), (Q, D), (Q + 1, D)])
                    else:
                        edges.append(D)
                names.append(f"P{n}_{m}")
                if fermi_system:
                    edges.append(Fedge[0, 1])  # F: out, T: in
                else:
                    edges.append(2)
                network[n][m] = Tensor(names, edges).randn()
        self.network = network
        self.environment = {"H": [[None for m in range(M - 1)] for n in range(N)], "V": [[None for m in range(M)] for n in range(N - 1)]}
        print("Lattice created")

    def get_energy(self):
        N = self.N
        M = self.M
        hamiltonian = self.hamiltonian
        state = self.get_state()
        name_pair = {(f"P{n}_{m}", f"P{n}_{m}") for m in range(M) for n in range(N)}
        if fermi_system:
            name_pair.add(("T", "T"))
        statestate = state.conjugate().contract(state, name_pair)
        Hstate = None
        for n in range(N):
            for m in range(M):
                if m != M - 1:
                    this = state.contract(hamiltonian, {(f"P{n}_{m}", "I0"), (f"P{n}_{m+1}", "I1")}).edge_rename({"O0": f"P{n}_{m}", "O1": f"P{n}_{m+1}"})
                    if Hstate is None:
                        Hstate = this
                    else:
                        Hstate += this
                if n != N - 1:
                    this = state.contract(hamiltonian, {(f"P{n}_{m}", "I0"), (f"P{n+1}_{m}", "I1")}).edge_rename({"O0": f"P{n}_{m}", "O1": f"P{n+1}_{m}"})
                    Hstate += this
        stateHstate = state.conjugate().contract(Hstate, name_pair)
        return float(stateHstate / statestate) / N / M

    def __absorb(self, tensor, n, m, direction, multiple=True):
        if direction == "L":
            if m == 0:
                return tensor
            env = self.environment["H"][n][m - 1]
            op_dir = "R"
        elif direction == "R":
            if m == self.M - 1:
                return tensor
            env = self.environment["H"][n][m]
            op_dir = "L"
        elif direction == "U":
            if n == 0:
                return tensor
            env = self.environment["V"][n - 1][m]
            op_dir = "D"
        elif direction == "D":
            if n == self.N - 1:
                return tensor
            env = self.environment["V"][n][m]
            op_dir = "U"
        if env is None:
            return tensor
        if multiple == False:
            env = env.map(lambda x: 0 if x == 0 else 1 / x)
        return tensor.contract(env, {(f"{direction}{n}_{m}", op_dir)}).edge_rename({direction: f"{direction}{n}_{m}"})

    def __update_two_nearest_site(self, op, n, m, direction):
        # print("updating", n, m, "with direction", direction)
        if direction == "H":
            # H
            tensor_1 = self.network[n][m]
            tensor_1 = self.__absorb(tensor_1, n, m, "L")
            tensor_1 = self.__absorb(tensor_1, n, m, "R")
            tensor_1 = self.__absorb(tensor_1, n, m, "U")
            tensor_1 = self.__absorb(tensor_1, n, m, "D")
            tensor_2 = self.network[n][m + 1]
            # tensor_2 = self.__absorb(tensor_2, n, m+1, "L")
            tensor_2 = self.__absorb(tensor_2, n, m + 1, "R")
            tensor_2 = self.__absorb(tensor_2, n, m + 1, "U")
            tensor_2 = self.__absorb(tensor_2, n, m + 1, "D")

            tensor_1_q, tensor_1 = tensor_1.qr('r', {f"P{n}_{m}", f"R{n}_{m}"}, "R", "L")
            tensor_2_q, tensor_2 = tensor_2.qr('r', {f"P{n}_{m+1}", f"L{n}_{m+1}"}, "L", "R")

            big = tensor_1.contract(tensor_2, {(f"R{n}_{m}", f"L{n}_{m+1}")}).contract(op, {(f"P{n}_{m}", "I0"), (f"P{n}_{m+1}", "I1")}).edge_rename({"O0": f"P{n}_{m}", "O1": f"P{n}_{m+1}"})
            U, s, V = big.svd({i for i in tensor_1.names if i != f"R{n}_{m}"}, f"R{n}_{m}", f"L{n}_{m+1}", "L", "R", self.D)
            self.environment["H"][n][m] = s / s.norm_max()

            tensor_1_q = self.__absorb(tensor_1_q, n, m, "L", False)
            tensor_1_q = self.__absorb(tensor_1_q, n, m, "U", False)
            tensor_1_q = self.__absorb(tensor_1_q, n, m, "D", False)
            self.network[n][m] = U.contract(tensor_1_q, {("L", "R")})
            tensor_2_q = self.__absorb(tensor_2_q, n, m + 1, "R", False)
            tensor_2_q = self.__absorb(tensor_2_q, n, m + 1, "U", False)
            tensor_2_q = self.__absorb(tensor_2_q, n, m + 1, "D", False)
            self.network[n][m + 1] = V.contract(tensor_2_q, {("R", "L")})

        else:
            # V
            tensor_1 = self.network[n][m]
            tensor_1 = self.__absorb(tensor_1, n, m, "L")
            tensor_1 = self.__absorb(tensor_1, n, m, "R")
            tensor_1 = self.__absorb(tensor_1, n, m, "U")
            tensor_1 = self.__absorb(tensor_1, n, m, "D")
            tensor_2 = self.network[n + 1][m]
            tensor_2 = self.__absorb(tensor_2, n + 1, m, "L")
            tensor_2 = self.__absorb(tensor_2, n + 1, m, "R")
            # tensor_2 = self.__absorb(tensor_2, n+1, m, "U")
            tensor_2 = self.__absorb(tensor_2, n + 1, m, "D")

            tensor_1_q, tensor_1 = tensor_1.qr('r', {f"P{n}_{m}", f"D{n}_{m}"}, "D", "U")
            tensor_2_q, tensor_2 = tensor_2.qr('r', {f"P{n+1}_{m}", f"U{n+1}_{m}"}, "U", "D")

            big = tensor_1.contract(tensor_2, {(f"D{n}_{m}", f"U{n+1}_{m}")}).contract(op, {(f"P{n}_{m}", "I0"), (f"P{n+1}_{m}", "I1")}).edge_rename({"O0": f"P{n}_{m}", "O1": f"P{n+1}_{m}"})
            U, s, V = big.svd({i for i in tensor_1.names if i != f"D{n}_{m}"}, f"D{n}_{m}", f"U{n+1}_{m}", "U", "D", self.D)
            self.environment["V"][n][m] = s / s.norm_max()

            tensor_1_q = self.__absorb(tensor_1_q, n, m, "L", False)
            tensor_1_q = self.__absorb(tensor_1_q, n, m, "R", False)
            tensor_1_q = self.__absorb(tensor_1_q, n, m, "U", False)
            self.network[n][m] = U.contract(tensor_1_q, {("U", "D")})
            tensor_2_q = self.__absorb(tensor_2_q, n + 1, m, "L", False)
            tensor_2_q = self.__absorb(tensor_2_q, n + 1, m, "R", False)
            tensor_2_q = self.__absorb(tensor_2_q, n + 1, m, "D", False)
            self.network[n + 1][m] = V.contract(tensor_2_q, {("D", "U")})

    def get_state(self):
        N = self.N
        M = self.M
        nets = self.network
        envs = self.environment
        res = None
        for n in range(N):
            for m in range(M):
                t = nets[n][m]
                if res is None:
                    res = t
                else:
                    pairs = set()
                    if n != 0:
                        # up
                        pairs.add((f"D{n-1}_{m}", f"U{n}_{m}"))
                        env = envs["V"][n - 1][m]
                        if env is not None:
                            t = t.contract(env, {(f"U{n}_{m}", "D")}).edge_rename({"U": f"U{n}_{m}"})
                    if m != 0:
                        # left
                        pairs.add((f"R{n}_{m-1}", f"L{n}_{m}"))
                        env = envs["H"][n][m - 1]
                        if env is not None:
                            t = t.contract(env, {(f"L{n}_{m}", "R")}).edge_rename({"L": f"L{n}_{m}"})
                    res = res.contract(t, pairs)
        return res

    def update(self, T, S, D=None):
        if D is not None:
            self.D = D
        op = (self.hamiltonian * (-S)).exponential({("I0", "O0"), ("I1", "O1")})
        N = self.N
        M = self.M

        def site_iterator():
            for n in range(0, N):
                for m in range(0, M - 1, 2):
                    yield (n, m, "H")
            for n in range(0, N):
                for m in range(1, M - 1, 2):
                    yield (n, m, "H")
            for n in range(0, N - 1, 2):
                for m in range(0, M):
                    yield (n, m, "V")
            for n in range(1, N - 1, 2):
                for m in range(0, M):
                    yield (n, m, "V")

        # print("Start updating")
        for t in range(T):
            for n, m, direction in site_iterator():
                self.__update_two_nearest_site(op, n, m, direction)
            for n, m, direction in reversed(list(site_iterator())):
                self.__update_two_nearest_site(op, n, m, direction)
            # print("Step", t, "Energy", self.get_energy())
            print("Step", t, end='\r')

        print(self.D, self.get_energy())


square = Square(N=4, M=4, P=8, D=4)

square.update(T=1000, S=0.01, D=4)
square.update(T=1000, S=0.01, D=5)
square.update(T=1000, S=0.01, D=6)
square.update(T=1000, S=0.01, D=7)
square.update(T=1000, S=0.01, D=8)
square.update(T=1000, S=0.01, D=7)
square.update(T=1000, S=0.01, D=6)
square.update(T=1000, S=0.01, D=7)
square.update(T=1000, S=0.01, D=8)
square.update(T=1000, S=0.01, D=9)
square.update(T=1000, S=0.01, D=10)
square.update(T=1000, S=0.01, D=9)
square.update(T=1000, S=0.01, D=8)
square.update(T=1000, S=0.01, D=9)
square.update(T=1000, S=0.01, D=10)
square.update(T=1000, S=0.01, D=11)
square.update(T=1000, S=0.01, D=12)
square.update(T=1000, S=0.01, D=11)
square.update(T=1000, S=0.01, D=10)
square.update(T=1000, S=0.01, D=11)
square.update(T=1000, S=0.01, D=12)
square.update(T=1000, S=0.01, D=13)
square.update(T=1000, S=0.01, D=14)
square.update(T=1000, S=0.01, D=13)
square.update(T=1000, S=0.01, D=12)
square.update(T=1000, S=0.01, D=13)
square.update(T=1000, S=0.01, D=14)
square.update(T=1000, S=0.01, D=15)
square.update(T=1000, S=0.01, D=16)
square.update(T=1000, S=0.01, D=15)
square.update(T=1000, S=0.01, D=14)
square.update(T=1000, S=0.01, D=15)
square.update(T=1000, S=0.01, D=16)
square.update(T=1000, S=0.01, D=17)
square.update(T=1000, S=0.01, D=18)
square.update(T=1000, S=0.01, D=17)
square.update(T=1000, S=0.01, D=16)
square.update(T=1000, S=0.01, D=17)
square.update(T=1000, S=0.01, D=18)
square.update(T=1000, S=0.01, D=19)
square.update(T=1000, S=0.01, D=20)
square.update(T=1000, S=0.01, D=19)
square.update(T=1000, S=0.01, D=18)
