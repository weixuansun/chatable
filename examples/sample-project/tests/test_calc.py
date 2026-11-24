from sample_project.calc import add, is_even


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_is_even():
    assert is_even(2) is True
    assert is_even(3) is False
    assert is_even(0) is True
