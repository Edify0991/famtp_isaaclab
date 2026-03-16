"""List FaMTP task ids registered in Gymnasium."""

from __future__ import annotations

import famtp_lab.tasks  # noqa: F401
from famtp_lab.utils.registry import registered_famtp_tasks


def main() -> None:
    """Print all project task ids."""
    task_ids = registered_famtp_tasks()
    if not task_ids:
        print("No FaMTP tasks found in Gymnasium registry.")
        return
    print("Registered FaMTP tasks:")
    for task_id in task_ids:
        print(f"- {task_id}")


if __name__ == "__main__":
    main()
