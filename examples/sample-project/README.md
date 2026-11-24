# sample-project（chatable 内置示例）

一个很小的 Python 包，用来体验「交给 agent」：

- `sample_project/calc.py` 里有两个函数，其中 `is_even` 有一个故意留下的 bug。
- `tests/test_calc.py` 是对应的 pytest 测试，当前 `test_is_even` 会失败。

试一试：让 agent「修复这个示例项目里失败的测试，让 pytest 全部通过」。
