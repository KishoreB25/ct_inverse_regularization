"""
test_astra.py
Quick diagnostic script to verify whether the ASTRA-toolbox is available,
or whether the native scikit-image and SciPy CT solvers are active.
"""

import sys


def test_astra():
    print("Testing ASTRA-toolbox and CT reconstruction environment...")
    print(f"Python: {sys.version.split()[0]}")

    try:
        import astra
        print(f"[OK] ASTRA Toolbox is installed (v{getattr(astra, '__version__', 'unknown')}).")
        astra.test()
        print("[OK] ASTRA test completed successfully.")
    except ImportError:
        print("[INFO] ASTRA Toolbox is not installed.")
        print("       The pipeline uses native scikit-image & SciPy CPU operators (no ASTRA required).")
        print("       To install ASTRA with GPU support: conda install -c astra-toolbox astra-toolbox")
    except Exception as e:
        print(f"[NOTE] ASTRA was detected but test returned: {e}")


if __name__ == "__main__":
    test_astra()
