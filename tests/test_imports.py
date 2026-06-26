"""Test imports for all modules in the bobiac_tools package."""

from __future__ import annotations

import sys


def test_all_package_functions():
    """Test that all functions defined in __all__ can be imported and are callable."""
    try:
        import bobiac_tools

        # Ensure the package has __all__ defined
        assert hasattr(bobiac_tools, "__all__"), "Package should define __all__"
        assert len(bobiac_tools.__all__) > 0, "Package __all__ should not be empty"

        # Test each function in __all__
        for func_name in bobiac_tools.__all__:
            # Get the function from the package
            func = getattr(bobiac_tools, func_name, None)

            # Verify it exists
            assert func is not None, (
                f"Function '{func_name}' listed in __all__ but not available"
            )

            # Verify it's callable
            assert callable(func), (
                f"'{func_name}' should be callable but is {type(func)}"
            )

            print(f"✓ Successfully imported and verified: {func_name}")

    except ImportError as e:
        raise AssertionError(f"Failed to import bobiac_tools package: {e}") from e


def test_package_dependencies():
    """Test that all required dependencies are available."""
    dependencies = ["numpy", "matplotlib"]

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ Dependency available: {dep}")
        except ImportError as e:
            raise AssertionError(
                f"Required dependency '{dep}' not available: {e}"
            ) from e


def test_basic_functionality():
    """Test basic functionality of imported dependencies."""
    try:
        import numpy as np

        # Test basic numpy functionality
        arr = np.array([1, 2, 3])
        assert arr.shape == (3,), "NumPy basic functionality failed"

        import matplotlib.pyplot as plt

        assert plt is not None, "Matplotlib import failed"

        print("✓ Basic dependency functionality verified")

    except Exception as e:
        raise AssertionError(f"Basic functionality test failed: {e}") from e


if __name__ == "__main__":
    """Run all tests when script is executed directly."""
    test_functions = [
        test_all_package_functions,
        test_package_dependencies,
        test_basic_functionality,
    ]

    print("Running comprehensive import tests...")
    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
