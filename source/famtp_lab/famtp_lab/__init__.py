"""FaMTP Isaac Lab external project package."""

from . import motion as motion

try:
    from . import tasks as tasks
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency for non-sim scripts
    tasks = None

__all__ = ["tasks", "motion"]
