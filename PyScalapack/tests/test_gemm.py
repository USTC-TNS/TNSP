# -*- coding: utf-8 -*-
import pytest
import numpy as np
import PyScalapack

scalapack = PyScalapack("libscalapack.so")


@pytest.mark.mpi(min_size=4)
def test_pdgemm():
    L1 = 128
    L2 = 512
    with scalapack(b'C', 2, 2) as context, scalapack(b'C', 1, 1) as context0:
        if context:
            # Create array0 add 1*1 grid
            array0 = context0.array(m=L1, n=L2, mb=1, nb=1, dtype=np.float64)
            if context0:
                array0.data[...] = np.random.randn(*array0.data.shape)

            # Redistribute array0 to 2*2 grid as array
            array = context.array(m=L1, n=L2, mb=1, nb=1, dtype=np.float64)
            scalapack.pgemr2d["D"](*(L1, L2), *array0.scalapack_params(), *array.scalapack_params(), context.ictxt)

            # Call pdgemm to get the product of array and array in 2*2 grid
            result = context.array(m=L1, n=L1, mb=1, nb=1, dtype=np.float64)
            scalapack.pdgemm(
                b'N',
                b'T',
                *(L1, L1, L2),
                scalapack.d_one,
                *array.scalapack_params(),
                *array.scalapack_params(),
                scalapack.d_zero,
                *result.scalapack_params(),
            )

            # Redistribute result to 1*1 grid as result0
            result0 = context0.array(m=L1, n=L1, mb=1, nb=1, dtype=np.float64)
            scalapack.pgemr2d["D"](*(L1, L1), *result.scalapack_params(), *result0.scalapack_params(), context.ictxt)

            # Check result0 == array0 * array0^T
            if context0:
                diff = result0.data - array0.data @ array0.data.T
                assert np.linalg.norm(diff) < 1e-8


def test_dgemm():
    L1 = 128
    L2 = 512
    with scalapack(b'C', 1, 1) as context:
        if context:
            array = context.array(m=L1, n=L2, mb=1, nb=1, dtype=np.float64)
            array.data[...] = np.random.randn(*array.data.shape)

            result = context.array(m=L1, n=L1, mb=1, nb=1, dtype=np.float64)
            scalapack.dgemm(
                b'N',
                b'T',
                *(L1, L1, L2),
                scalapack.d_one,
                *array.lapack_params(),
                *array.lapack_params(),
                scalapack.d_zero,
                *result.lapack_params(),
            )

            diff = result.data - array.data @ array.data.T
            assert np.linalg.norm(diff) < 1e-8
