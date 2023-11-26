import numpy as np
import TAT


def test_main():
    Tensor = TAT.No.D.Tensor

    max_random = 8

    for _ in range(1000):
        rank_A = np.random.randint(2, max_random)
        rank_contract = np.random.randint(1, rank_A)
        U_leg = np.random.choice(range(rank_A), rank_contract, False)

        dim_A = np.random.randint(1, max_random, size=rank_A)

        A = Tensor([f"A.{i}" for i in range(rank_A)], dim_A.tolist()).randn_()

        U, S, V = A.svd({f"A.{i}" for i in U_leg}, "SVD.U", "SVD.V", "S.U", "S.V")
        re_A = U.contract(S, {("SVD.U", "S.U")}).edge_rename({"S.V": "SVD.U"}).contract(V, {("SVD.U", "SVD.V")})
        diff = re_A - A

        UTU = U.contract(U.edge_rename({"SVD.U": "new"}), {(name, name) for name in U.names if name != "SVD.U"})
        UTU = UTU.blocks[UTU.names]
        VTV = V.contract(V.edge_rename({"SVD.V": "new"}), {(name, name) for name in V.names if name != "SVD.V"})
        VTV = VTV.blocks[VTV.names]

        diff_U = UTU - np.identity(len(UTU))
        diff_V = VTV - np.identity(len(VTV))

        assert np.max([diff.norm_max(), np.max(np.abs(diff_U)), np.max(np.abs(diff_V))]) < 1e-6
