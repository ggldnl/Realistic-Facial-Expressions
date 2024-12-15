import warnings
import functools

def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and may be removed in future versions.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)
    return wrapper

def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} has some bugs and needs to be fixed.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)
    return wrapper
