# -*- coding: utf-8 -*-
import TAT


def test_arrow_and_parity():
    assert TAT.arrow(+1) == False
    assert TAT.arrow(-1) == True
    assert TAT.parity(+1) == False
    assert TAT.parity(-1) == True
