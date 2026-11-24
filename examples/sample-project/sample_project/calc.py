"""Two tiny functions; one of them has a deliberate bug."""


def add(a, b):
    return a + b


def is_even(n):
    # BUG: this is True for odd numbers — it should be `n % 2 == 0`.
    return n % 2 == 1
