# SPDX-License-Identifier: Apache-2.0
"""The package is installed (pip install -e ".[dev]"); tests import motionmcp.client from it."""

import pytest
from fake_server import FakeMmcpServer


@pytest.fixture
def mmcp_server():
    """A running :class:`fake_server.FakeMmcpServer`; closed after the test."""
    server = FakeMmcpServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()
