# -*- coding: utf-8 -*-
import TAT


def test_no_symmetry_example_0():
    a = TAT.No.D.Tensor(["A", "B"], [8, 8]).range_()
    b = a.edge_rename({
        "A": "C"
    }).edge_operator({
        "C": [("D", 4), ("E", 2)],
        "B": [("F", 2), ("G", 4)],
    }, {"D", "F"}, {
        "I": ["D", "F"],
        "J": ["G", "E"],
    }, ["J", "I"])
    b_s = a.edge_rename({
        "A": "C"
    }).split_edge({
        "C": [("D", 4), ("E", 2)],
        "B": [("F", 2), ("G", 4)],
    }).merge_edge({
        "I": ["D", "F"],
        "J": ["G", "E"],
    }).transpose(["J", "I"])
    assert (b - b_s).norm_max() == 0


def test_u1_symmetry_example_0():
    a = TAT.U1.D.Tensor(["Left", "Right", "Up", "Down"], [
        [(-1, 3), (0, 1), (1, 2)],
        [(-1, 1), (0, 4), (1, 2)],
        [(-1, 2), (0, 3), (1, 1)],
        [(-1, 1), (0, 3), (1, 2)],
    ]).range_()
    b = a.edge_rename({
        "Right": "Right1"
    }).split_edge({"Down": [
        ("Down1", [(0, 1), (1, 2)]),
        ("Down2", [(-1, 1), (0, 1)]),
    ]})
    c = b.transpose(["Down1", "Right1", "Up", "Left", "Down2"])
    d = c.merge_edge({"Left": ["Left", "Down2"]})
    total = a.edge_rename({
        "Right": "Right1"
    }).edge_operator(
        {"Down": [("Down1", [(0, 1), (1, 2)]), ("Down2", [(-1, 1), (0, 1)])]},
        set(),
        {"Left": ["Left", "Down2"]},
        ["Down1", "Right1", "Up", "Left"],
    )
    assert (total - d).norm_max() == 0


def test_fermi_symmetry_example_0():
    a = TAT.Fermi.D.Tensor(["Left", "Right", "Up", "Down"], [
        [(-1, 3), (0, 1), (1, 2)],
        [(-1, 1), (0, 4), (1, 2)],
        [(-1, 2), (0, 3), (1, 1)],
        [(-1, 1), (0, 3), (1, 2)],
    ]).range_()
    b = a.edge_rename({
        "Right": "Right1"
    }).split_edge({"Down": [
        ("Down1", [(0, 1), (1, 2)]),
        ("Down2", [(-1, 1), (0, 1)]),
    ]})
    r = b.reverse_edge({"Left"})
    c = r.transpose(["Down1", "Right1", "Up", "Left", "Down2"])
    d = c.merge_edge({"Left": ["Left", "Down2"]})
    total = a.edge_rename({
        "Right": "Right1"
    }).edge_operator(
        {"Down": [("Down1", [(0, 1), (1, 2)]), ("Down2", [(-1, 1), (0, 1)])]},
        {"Left"},
        {"Left": ["Left", "Down2"]},
        ["Down1", "Right1", "Up", "Left"],
    )
    assert (total - d).norm_max() == 0
