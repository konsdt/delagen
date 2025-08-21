# deep_ela/__init__.py
__version__ = "0.1.0"
__all__ = ["load_deepela", "DeepELA"]

def __getattr__(name):
    if name == "load_deepela":
        # import only when actually requested
        from .inference import load_deepela as f
        return f
    if name == "DeepELA":
        from .inference import DeepELA
        return DeepELA
    raise AttributeError(name)

def __dir__():
    return sorted(list(globals().keys()) + __all__)