"""Exceptions raised by model loaders."""


class ModelUnavailableError(RuntimeError):
    """A model backend could not be loaded (missing weights, no network, no GPU, ...).

    Callers should catch this and either degrade to a documented fallback or
    surface the message to the user. The message always says *which* model and
    *why* so that fallbacks are never silent.
    """
