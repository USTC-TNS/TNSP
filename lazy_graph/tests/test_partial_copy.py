from lazy import Root, Node, Copy

calculate_count = 0


def add(a, b):
    global calculate_count
    calculate_count += 1
    return a + b


def test_copy_lazy_graph():
    a = Root(1)
    b = Root(2)
    c = Node(add, a, b)
    assert calculate_count == 0
    assert c() == 3
    assert calculate_count == 1

    copy = Copy()
    new_a = copy(a)
    new_c = copy(c)

    assert new_c() == 3
    assert calculate_count == 1
    a.reset(4)
    assert c() == 6
    assert calculate_count == 2
    assert new_c() == 3
    assert calculate_count == 2
    new_a.reset(8)
    assert c() == 6
    assert calculate_count == 2
    assert new_c() == 10
    assert calculate_count == 3

    b.reset(10)
    assert c() == 14
    assert calculate_count == 4
    assert new_c() == 18
    assert calculate_count == 5
