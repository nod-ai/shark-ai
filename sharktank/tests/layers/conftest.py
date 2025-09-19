def pytest_addoption(parser):
    """Add command line option for IRPA file path."""
    parser.addoption(
        "--irpa-path",
        action="store",
        default=None,
        help="Path to the IRPA file for testing"
    )
