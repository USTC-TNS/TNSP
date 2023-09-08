import numpy as np
import TAT


def test_main():
    Tensor = TAT.No.D.Tensor

    max_random = 8

    for _ in range(100):
        rank_A = np.random.randint(3, max_random)
        rank_trace = np.random.randint(1, rank_A // 2 + 1)
        pair_leg = np.random.choice(range(rank_A), [rank_trace, 2], False)

        name_list = [f"A.{i}" for i in range(rank_A)]
        dim_list = np.random.randint(2, max_random, rank_A)
        dim_trace = np.random.randint(2, max_random, rank_trace)
        for i, (j, k) in enumerate(pair_leg):
            dim_list[j] = dim_trace[i]
            dim_list[k] = dim_trace[i]

        trace_conf = {(f"A.{i}", f"A.{j}") for i, j in pair_leg}

        A = Tensor(name_list, dim_list.tolist()).range()
        B = A.trace(trace_conf)

        res = A.blocks[A.names]
        for i in range(rank_trace):
            res = res.trace(0, pair_leg[i, 0], pair_leg[i, 1])
            for j in range(i + 1, rank_trace):
                if pair_leg[j, 0] > pair_leg[i, 0]:
                    pair_leg[j, 0] -= 1
                if pair_leg[j, 1] > pair_leg[i, 0]:
                    pair_leg[j, 1] -= 1
                if pair_leg[i, 1] > pair_leg[i, 0]:
                    pair_leg[i, 1] -= 1
                if pair_leg[j, 0] > pair_leg[i, 1]:
                    pair_leg[j, 0] -= 1
                if pair_leg[j, 1] > pair_leg[i, 1]:
                    pair_leg[j, 1] -= 1

        diff = res - B.blocks[B.names]

        assert np.max(np.abs(diff)) < 1e-6
