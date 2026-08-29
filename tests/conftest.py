"""
Pytest configuration for conntrail tests.
"""
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load repo-root .env so integration tests have API keys available
load_dotenv(Path(__file__).parent.parent / ".env")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires live API keys")
    config.addinivalue_line("markers", "slow: takes > 10 seconds")


def pytest_sessionfinish(session, exitstatus):
    # Treat "no tests collected" as success on the empty skeleton (no tests
    # exist yet). Once real tests land this branch is never hit.
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
