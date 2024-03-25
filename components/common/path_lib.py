"""This module contains methods for paths managing."""
from pathlib import Path
from typing import Optional


def get_absolute_path(relative_path: str,
                      root_dir: Optional[Path] = None) -> Path:
    """Returns correct absolute path as Path object

    Args:
        path: A relative path.
        root_dir: An absolute path to the root folder.

    Returns:
        An absolute path.
    """
    parts = relative_path.split('/')
    res_path = Path(*parts)
    if root_dir is not None:
        res_path = Path(root_dir) / res_path
    return res_path
