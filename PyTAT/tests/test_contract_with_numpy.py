# -*- coding: utf-8 -*-
import numpy as np
import TAT


def test_main():
    max_random = 8
    Tensor = TAT.No.D.Tensor

    for _ in range(100):
        rank_A = np.random.randint(2, max_random)
        rank_B = np.random.randint(2, max_random)
        rank_contract = np.random.randint(1, np.min([rank_A, rank_B]))
        # print(rank_A, rank_B, rank_contract)

        contract_name_A = np.random.choice(range(rank_A), rank_contract, False)
        contract_name_B = np.random.choice(range(rank_B), rank_contract, False)

        dim_A = np.random.randint(1, max_random, size=rank_A)
        dim_B = np.random.randint(1, max_random, size=rank_B)
        dim_contract = np.random.randint(1, max_random, size=rank_contract)

        dim_A = [
            int(j if i not in contract_name_A else dim_contract[contract_name_A.tolist().index(i)])
            for i, j in enumerate(dim_A)
        ]
        dim_B = [
            int(j if i not in contract_name_B else dim_contract[contract_name_B.tolist().index(i)])
            for i, j in enumerate(dim_B)
        ]

        A = Tensor([f"A.{i}" for i in range(rank_A)], dim_A).randn_()
        B = Tensor([f"B.{i}" for i in range(rank_B)], dim_B).randn_()
        v_t = A.contract(B, {(f"A.{i}", f"B.{j}") for i, j in zip(contract_name_A, contract_name_B)})
        v_t = v_t.blocks[v_t.names]
        v_n = np.tensordot(A.blocks[A.names], B.blocks[B.names], [contract_name_A, contract_name_B])
        v_d = v_t - v_n
        assert np.max(np.abs(v_d)) < 1e-6
