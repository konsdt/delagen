# deep_ela/__init__.py
__version__ = "0.1.0"
__all__ = ["load_deepela", "DeepELA", "create_initial_sample"]

def __getattr__(name):
    if name == "load_deepela":
        # import only when actually requested
        from .inference import load_deepela as f
        return f
    if name == "DeepELA":
        from .inference import DeepELA
        return DeepELA
    if name == "create_initial_sample":
        from .sampling import create_initial_sample
        return create_initial_sample
    raise AttributeError(name)

def __dir__():
    return sorted(list(globals().keys()) + __all__)