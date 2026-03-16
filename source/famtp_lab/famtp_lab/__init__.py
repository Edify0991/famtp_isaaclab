"""FaMTP Isaac Lab external project package."""

from . import motion as motion

try:
    from . import priors as priors
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    priors = None

try:
    from . import baselines as baselines
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    baselines = None

try:
    from . import tasks as tasks
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency for non-sim scripts
    tasks = None

__all__ = ["tasks", "motion", "priors", "baselines"]
