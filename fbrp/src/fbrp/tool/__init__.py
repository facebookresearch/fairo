import fbrp
import os
import sys
from importlib.machinery import SourceFileLoader


def main():
    # Default the entrypoint file to the fsetup.py in the current directory.
    fsetup_path = os.path.abspath("fsetup.py")
    # The user may specify another entrypoint with:
    # fbrp -f <file> [args...]
    if len(sys.argv) >= 3 and sys.argv[1] == "-f":
        fsetup_path = os.path.abspath(sys.argv[2])
        sys.argv = sys.argv[:1] + sys.argv[3:]

    # Show help even if there is no fsetup.py.
    if not os.path.exists(fsetup_path):
        fbrp.main()
        return

    # Run the entrypoint file.
    sys.path.append(os.path.dirname(fsetup_path))
    SourceFileLoader("fsetup", fsetup_path).load_module()


if __name__ == "__main__":
    main()
