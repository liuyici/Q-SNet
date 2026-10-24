"""Compatibility entry point for the fixed-schedule LOSO-UDA protocol.

The former development script selected iterations using target emotion labels.
Use Git history for that historical code; it is not a valid evaluation protocol.
Both this command and main.py now train without target emotion labels and
evaluate the final checkpoint once. See ../docs/reproducibility.md.
"""

from main import main


if __name__ == "__main__":
    main()
