import pathlib
import re


def glob_level_environments(
    directory: str | pathlib.Path, pattern: str = "level*.pkl"
) -> dict[int, pathlib.Path]:
    """
    Find all pickle files matching a pattern and extract the level number.

    Parameters
    ----------
    directory : str | pathlib.Path
        Directory to search within.
    pattern : str
        Glob pattern to match files (default: "level*.pkl").

    Returns
    -------
    dict[int, pathlib.Path]
        List of (level, path) pairs discovered in the directory.
    """
    if isinstance(directory, str):
        directory = pathlib.Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    matches: dict[int, pathlib.Path] = {}

    for p in directory.glob(pattern):
        m = re.search(r"level(\d+)", p.stem)
        if not m:
            continue
        level = int(m.group(1))
        matches[level] = p

    return matches
