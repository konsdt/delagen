def test_import_smoke():
    # ensure the module actually got loaded
    import importlib, sys
    m = importlib.import_module("deep_ela")
    assert "deep_ela" in sys.modules

    # ensure it looks like a proper module
    assert m.__spec__ is not None

    assert callable(getattr(m, "create_initial_sample", None))
    assert callable(getattr(m, "load_deepela", None))
