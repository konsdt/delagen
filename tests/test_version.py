def test_package_has_version_metadata():
    # Either expose deep_ela.__version__ OR have project metadata installed
    import importlib
    from importlib.metadata import PackageNotFoundError, version

    pkg = importlib.import_module("deep_ela")

    has_dunder = getattr(pkg, "__version__", None) is not None
    try:
        has_metadata = version("deep_ela") is not None
    except PackageNotFoundError:
        has_metadata = False

    assert has_dunder or has_metadata, (
        "Expose __version__ in deep_ela/__init__.py or ensure the package "
        "metadata is available when installed."
    )
