import pytest
import PyScalapack
import numpy as np

scalapack = PyScalapack("libscalapack.so")


@pytest.mark.mpi(min_size=2)
def test_array_redistribute_0():
    # matrix:
    # 1 2
    # 3 4
    with scalapack(b'R', 1, 2) as context1, scalapack(b'R', 2, 1) as context2:
        if context1:
            m = 2
            n = 2
            array1 = context1.array(m=m, n=n, mb=1, nb=1, dtype=np.float64)
            if context1.rank.value == 0:
                array1.data[0, 0] = 1
                array1.data[1, 0] = 3
            else:
                array1.data[0, 0] = 2
                array1.data[1, 0] = 4
            array2 = context2.array(m=m, n=n, mb=1, nb=1, dtype=np.float64)
            scalapack.pgemr2d["D"](
                *(m, n),
                *array1.scalapack_params(),
                *array2.scalapack_params(),
                context1.ictxt,
            )
            if context1.rank.value == 0:
                assert array2.data[0, 0] == 1
                assert array2.data[0, 1] == 2
            else:
                assert array2.data[0, 0] == 3
                assert array2.data[0, 1] == 4


@pytest.mark.mpi(min_size=4)
def test_array_redistribute_1():
    with scalapack(b'R', 2, 2) as context1, scalapack(b'R', 1, 1) as context0:
        if context1:
            m = 100
            n = 100
            array0 = context0.array(m=m, n=n, mb=1, nb=1, dtype=np.float64)
            array2 = context0.array(m=m, n=n, mb=1, nb=1, dtype=np.float64)
            if context0:
                array0.data[...] = np.random.randn(*array0.data.shape)
            array1 = context1.array(m=m, n=n, mb=1, nb=1, dtype=np.float64)
            scalapack.pgemr2d["D"](
                *(m, n),
                *array0.scalapack_params(),
                *array1.scalapack_params(),
                context1.ictxt,
            )
            scalapack.pgemr2d["D"](
                *(m, n),
                *array1.scalapack_params(),
                *array2.scalapack_params(),
                context1.ictxt,
            )
            if context1:
                assert np.linalg.norm(array2.data - array0.data) == 0
