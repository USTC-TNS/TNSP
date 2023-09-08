import numpy as np
import TAT


def test_main():
    max_random = 8
    Tensor = TAT.No.D.Tensor

    for _ in range(100):
        rank_A = np.random.randint(3, max_random)
        rank_B = np.random.randint(3, max_random)
        rank_total = np.random.randint(2, np.min([rank_A, rank_B]))
        rank_contract = np.random.randint(1, rank_total)
        rank_fuse = rank_total - rank_contract

        total_leg_A = np.random.choice(range(rank_A), rank_total, False)
        total_leg_B = np.random.choice(range(rank_B), rank_total, False)
        contract_leg_A = total_leg_A[:rank_contract]
        contract_leg_B = total_leg_B[:rank_contract]
        fuse_leg_A = total_leg_A[rank_contract:]
        fuse_leg_B = total_leg_B[rank_contract:]

        name_list_A = [f"A.{i}" for i in range(rank_A)]
        name_list_B = [f"B.{i}" for i in range(rank_B)]
        fuse_names = set()
        for i in range(rank_fuse):
            name = f"C.{fuse_leg_B[i]}"
            name_list_A[fuse_leg_A[i]] = name
            name_list_B[fuse_leg_B[i]] = name
            fuse_names.add(name)

        dim_A = np.random.randint(1, max_random, rank_A).tolist()
        dim_B = np.random.randint(1, max_random, rank_B).tolist()
        for i in range(rank_total):
            dim_A[total_leg_A[i]] = dim_B[total_leg_B[i]] = np.random.randint(2, max_random)

        A = Tensor(name_list_A, dim_A).range()
        B = Tensor(name_list_B, dim_B).range()
        C = A.contract(B, {(f"A.{contract_leg_A[i]}", f"B.{contract_leg_B[i]}") for i in range(rank_contract)},
                       fuse_names)
        # print(repr(A))
        # print(repr(B))
        # print(repr(C))
        a = A.blocks[A.names]
        b = B.blocks[B.names]

        index_A = [chr(ord('a') + i) for i in range(rank_A)]
        index_B = [chr(ord('A') + i) for i in range(rank_B)]
        index_C = []
        for i in range(rank_total):
            index_A[total_leg_A[i]] = index_B[total_leg_B[i]]
        for c_name in C.names:
            if c_name.startswith("A"):
                index_C.append(chr(ord('a') + int(c_name[2:])))
            elif c_name.startswith("B"):
                index_C.append(chr(ord('A') + int(c_name[2:])))
            else:
                index_C.append(chr(ord('A') + int(c_name[2:])))
        ein_conf = "".join(index_A) + "," + "".join(index_B) + "->" + "".join(index_C)
        # print(ein_conf)
        c = np.einsum(ein_conf, a, b)
        # print(c.shape)
        assert np.max(np.abs(C.blocks[C.names] - c)) < 1e-6
