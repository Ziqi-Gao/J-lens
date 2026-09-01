"""Errors shared by the project-side ServerScheduler integration."""


class ServerSchedulerAdapterError(RuntimeError):
    """A central manifest, dispatch request, or completion gate is invalid."""
