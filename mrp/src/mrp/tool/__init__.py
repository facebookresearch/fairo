import mrp
import os
import sys
from importlib.machinery import SourceFileLoader


def main():
    # Default the entrypoint file to the msetup.py in the current directory.
    msetup_path = os.path.abspath("msetup.py")
    # The user may specify another entrypoint with:
    # mrp -f <file> [args...]
    if len(sys.argv) >= 3 and sys.argv[1] == "-f":
        msetup_path = os.path.abspath(sys.argv[2])
        sys.argv = sys.argv[:1] + sys.argv[3:]

    # Show help even if there is no msetup.py.
    if not os.path.exists(msetup_path):
        mrp.main()
        return

    # Run the entrypoint file.
    sys.path.append(os.path.dirname(msetup_path))
    SourceFileLoader("msetup", msetup_path).load_module()


if __name__ == "__main__":
    main()
