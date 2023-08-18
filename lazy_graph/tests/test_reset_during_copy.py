from lazy import Root, Node, Copy

calculate_count = 0


def add(a, b):
    global calculate_count
    calculate_count += 1
    return a + b


def test_reset_during_copy():
    a = Root(1)
    b = Root(2)
    c = Node(add, a, b)
    assert c() == 3

    copy = Copy()
    new_a = copy(a)
    new_b = copy(b)
    new_c = copy(c)
    assert bool(new_c) is True

    copy2 = Copy()
    new_a2 = copy2(a)
    new_b2 = copy2(b)
    new_a2.reset(10)
    new_c2 = copy2(c)
    assert bool(new_c2) is False
