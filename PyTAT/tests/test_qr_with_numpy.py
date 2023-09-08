import numpy as np
import TAT


def test_main():
    Tensor = TAT.No.D.Tensor

    max_random = 8

    for _ in range(100):
        rank_A = np.random.randint(2, max_random)
        rank_contract = np.random.randint(1, rank_A)
        U_leg = np.random.choice(range(rank_A), rank_contract, False)

        dim_A = np.random.randint(1, max_random, size=rank_A)

        A = Tensor([f"A.{i}" for i in range(rank_A)], dim_A.tolist()).randn()

        Q, R = A.qr("Q", {f"A.{i}" for i in U_leg}, "QR.Q", "QR.R")
        re_A = Q.contract(R, {("QR.Q", "QR.R")})
        diff = re_A - A

        QTQ = Q.contract(Q.edge_rename({"QR.Q": "new"}), {(name, name) for name in Q.names if name != "QR.Q"})
        QTQ = QTQ.blocks[QTQ.names]

        diff_Q = QTQ - np.identity(len(QTQ))

        assert np.max([diff.norm_max(), np.max(np.abs(diff_Q))]) < 1e-6
        R_block = R.blocks[R.names]
        # print(R_block.shape)
        # print(R_block.reshape([-1, R_block.shape[-1]]))
        # print(R_block.reshape([R_block.shape[0], -1]))
