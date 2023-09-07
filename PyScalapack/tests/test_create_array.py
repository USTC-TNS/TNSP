import pytest
import PyScalapack
import numpy as np

scalapack = PyScalapack("libscalapack.so")


@pytest.mark.mpi(min_size=4)
def test_array_create_incorrect():
    with scalapack(b'C', nprow=2, npcol=2) as context:
        with pytest.raises(RuntimeError) as e_info:
            array = context.array(m=10, n=10, mb=1, nb=1)
        with pytest.raises(RuntimeError) as e_info:
            array = context.array(m=10, n=10, mb=1, nb=1, dtype=np.float64, data=np.zeros([5, 5], dtype=np.float64))


@pytest.mark.mpi(min_size=4)
def test_array_create_own_data():
    with scalapack(b'C', nprow=2, npcol=2) as context:
        array = context.array(m=10, n=10, mb=1, nb=1, dtype=np.float64)
        assert array.dtype == array.c_dtype.value == 1
        assert array.ctxt == array.c_ctxt.value == context.ictxt.value
        assert array.m == array.c_m.value == 10
        assert array.n == array.c_n.value == 10
        assert array.mb == array.c_mb.value == 1
        assert array.nb == array.c_nb.value == 1
        assert array.rsrc == array.c_rsrc.value == 0
        assert array.csrc == array.c_csrc.value == 0
        if context:
            assert array.lld == array.c_lld.value == 5


@pytest.mark.mpi(min_size=4)
def test_array_create_share_data():
    with scalapack(b'C', nprow=2, npcol=2) as context:
        array = context.array(m=10, n=10, mb=1, nb=1, data=np.zeros([5, 5], dtype=np.float64, order='F'))
        with pytest.raises(RuntimeError) as e_info:
            array = context.array(m=10, n=10, mb=1, nb=1, data=np.zeros([5, 5], dtype=np.float64, order='C'))

    with scalapack(b'R', nprow=2, npcol=2) as context:
        array = context.array(m=10, n=10, mb=1, nb=1, data=np.zeros([5, 5], dtype=np.float64, order='C'))
        with pytest.raises(RuntimeError) as e_info:
            array = context.array(m=10, n=10, mb=1, nb=1, data=np.zeros([5, 5], dtype=np.float64, order='F'))
        #with pytest.raises(RuntimeError) as e_info:
        #    array = context.array(m=10, n=10, mb=1, nb=1, data=np.zeros([6, 6], dtype=np.float64, order='C'))


@pytest.mark.mpi(min_size=4)
def test_array_create_share_data_local_mismatch():
    with scalapack(b'C', nprow=2, npcol=2) as context:
        with pytest.raises(RuntimeError) as e_info:
            array = context.array(m=10, n=10, mb=1, nb=1, data=np.zeros([6, 6], dtype=np.float64, order='F'))
            if not context:
                raise RuntimeError("process in context should raise, process out context should not, raise it manually")
