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
    assert c() == 3

    copy = Copy()
    new_a = copy(a)
    new_b = copy(b)
    new_c = copy(c)
    new_c2 = copy(c)
    assert id(new_c) == id(new_c2)
