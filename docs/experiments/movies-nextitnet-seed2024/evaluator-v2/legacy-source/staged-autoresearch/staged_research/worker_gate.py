"""Private process launch gate: never train before the parent commits its PID."""
import os
import sys

if __name__ == "__main__":
    # Avoid buffered read-ahead: the remaining stdin is the native-agent prompt.
    if os.read(0, 1) != b"\n":
        sys.exit(125)
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
