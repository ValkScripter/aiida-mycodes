"""pytest fixtures for simplified testing."""

import pytest

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]


@pytest.fixture(scope="function", autouse=True)
def clear_database_auto(clear_database):  # pylint: disable=unused-argument
    """Automatically clear database in between tests."""


@pytest.fixture(scope="function")
def mycodes_code(aiida_local_code_factory):
    """Get a mycodes code."""
    return aiida_local_code_factory(executable="diff", entry_point="mycodes.diff")


@pytest.fixture(scope="function")
def mycodes_pw(aiida_local_code_factory):
    """Get a mycodes pw code."""
    return aiida_local_code_factory(executable="bash", entry_point="mycodes.pw.base")
