"""
Smoke test will confirms the package imports and pytest discovery works.

The brief assesses software engineering and testing across all grade bands;
for a 1st, testing must be evident in *all* components. Real tests will
replace this one as each module is built.
"""

import src


def test_package_imports() -> None:
    """The top-level package should expose a version string."""
    assert hasattr(src, "__version__")
    assert isinstance(src.__version__, str)
    assert len(src.__version__) > 0


def test_subpackages_importable() -> None:
    """Every planned subpackage must import cleanly."""
    from src import backtest, data, explain, features, models, web  # noqa: F401
