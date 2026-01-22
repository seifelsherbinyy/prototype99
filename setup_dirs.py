"""Initialize project directory structure. Idempotent; safe to run repeatedly."""

from pathlib import Path

DIRS = [
    ("data/selection", "Reserved for specific data variants"),
    ("data/dropzone", "Raw multi-file ingestion and historical archives"),
    ("src", "Modular logic"),
]


def ensure_dirs() -> None:
    """Create project directories if missing. No output. Safe for app startup."""
    root = Path(__file__).resolve().parent
    for rel_path, _ in DIRS:
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    for rel_path, purpose in DIRS:
        path = root / rel_path
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        status = "exists" if existed else "created"
        print(f"  {rel_path}: {status} ({purpose})")


if __name__ == "__main__":
    print("Project directories:")
    main()
    print("Done.")
