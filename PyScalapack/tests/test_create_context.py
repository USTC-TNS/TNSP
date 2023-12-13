# -*- coding: utf-8 -*-
import pytest
import PyScalapack

scalapack = PyScalapack("libscalapack.so")


@pytest.mark.mpi(min_size=4)
def test_context_column_major():
    with scalapack(b'C', nprow=2, npcol=2) as context:
        assert context.layout.value == b'C'
        if context:
            assert context.rank.value // context.nprow.value == context.mycol.value
            assert context.rank.value % context.nprow.value == context.myrow.value
            assert context.rank.value < context.nprow.value * context.npcol.value
        else:
            assert context.rank.value >= context.nprow.value * context.npcol.value


@pytest.mark.mpi(min_size=4)
def test_context_row_major():
    with scalapack(b'R', nprow=1, npcol=2) as context:
        assert context.layout.value == b'R'
        if context:
            assert context.rank.value % context.npcol.value == context.mycol.value
            assert context.rank.value // context.npcol.value == context.myrow.value
            assert context.rank.value < context.nprow.value * context.npcol.value
        else:
            assert context.rank.value >= context.nprow.value * context.npcol.value


def test_context_error_major():
    with pytest.raises(RuntimeError):
        with scalapack(b'W', nprow=2, npcol=2) as context:
            pass


@pytest.mark.mpi(min_size=2)
def test_context_auto_row():
    with scalapack(b'R', nprow=-1, npcol=2) as context:
        assert context.nprow.value == context.size.value // context.npcol.value
        assert context.layout.value == b'R'
        if context:
            assert context.rank.value % context.npcol.value == context.mycol.value
            assert context.rank.value // context.npcol.value == context.myrow.value
            assert context.rank.value < context.nprow.value * context.npcol.value
        else:
            assert context.rank.value >= context.nprow.value * context.npcol.value


@pytest.mark.mpi(min_size=2)
def test_context_auto_column():
    with scalapack(b'R', nprow=2, npcol=-1) as context:
        assert context.npcol.value == context.size.value // context.nprow.value
        assert context.layout.value == b'R'
        if context:
            assert context.rank.value % context.npcol.value == context.mycol.value
            assert context.rank.value // context.npcol.value == context.myrow.value
            assert context.rank.value < context.nprow.value * context.npcol.value
        else:
            assert context.rank.value >= context.nprow.value * context.npcol.value


@pytest.mark.mpi(min_size=4)
def test_context_barrier():
    with scalapack(b'R', nprow=2, npcol=2) as context:
        context.barrier()
        context.barrier(b'A')
        context.barrier(b'R')
        context.barrier(b'C')
        context.barrier(scope=b'A')
        context.barrier(scope=b'R')
        context.barrier(scope=b'C')

        with pytest.raises(RuntimeError):
            context.barrier(b'W')
        with pytest.raises(RuntimeError):
            context.barrier(scope=b'W')


def test_context_raise():
    with pytest.raises(RuntimeError) as e_info:
        with scalapack(b'R', nprow=1, npcol=1) as context:
            raise RuntimeError("Test Error")
    assert e_info.value.args == ("Test Error",)
