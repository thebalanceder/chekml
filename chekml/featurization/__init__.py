# chekml/featurization/__init__.py

__all__ = [
    "InequalityFeaturizerSlow",
    "InformationRepurposedFeaturizerSlow",
    "InformationRepurposedWrapper",
    "InformationRepurposedWrapperFast",
    "InequalityFeaturizerFast",
    "InformationRepurposedFeaturizerFast",
    "MetaheuristicFeaturizer",
    "FeatureEvaluator"
]

# --------------------------
# Lazy attribute loader (proxy objects)
# --------------------------


class _LazyImport:
    """Proxy that delays importing the real object until first use.

    Example usage:
        Proxy = _LazyImport('IF.slow.inequality_based_featurization', 'InequalityFeaturizer')
        # No heavy imports executed yet
        cls = Proxy  # still proxy
        model = cls(...)  # real module imported here
    """

    def __init__(self, module_subpath, attr_name):
        # module_subpath is relative to this package, e.g. 'IF.slow.inequality_based_featurization'
        self._module_path = f"{__name__}.{module_subpath}"
        self._attr_name = attr_name
        self._obj = None

    def _load(self):
        if self._obj is None:
            mod = __import__(self._module_path, fromlist=[self._attr_name])
            self._obj = getattr(mod, self._attr_name)

    def __call__(self, *args, **kwargs):
        self._load()
        return self._obj(*args, **kwargs)

    def __getattr__(self, item):
        self._load()
        return getattr(self._obj, item)

    def __repr__(self):
        return f"<LazyImport proxy for {self._module_path}.{self._attr_name}>"


# Mapping of exported names -> (module_subpath, attribute_name)
_LAZY_MAP = {
    "InequalityFeaturizerSlow": ("IF.slow.inequality_based_featurization", "InequalityFeaturizer"),
    "InequalityFeaturizerFast": ("IF.fast.inequality_based_featurization", "InequalityFeaturizer"),
    "InformationRepurposedFeaturizerSlow": ("IRF.slow.information_repurposed_featurization", "InformationRepurposedFeaturizer"),
    "InformationRepurposedFeaturizerFast": ("IRF.fast.information_repurposed_featurization", "InformationRepurposedFeaturizer"),
    "InformationRepurposedWrapper": ("IRF.slow.information_repurposed_featurization", "InformationRepurposedFeaturizerWrapper"),
    "InformationRepurposedWrapperFast": ("IRF.fast.information_repurposed_featurization", "InformationRepurposedFeaturizerWrapper"),
    "MetaheuristicFeaturizer": ("MhF.MhF", "MetaheuristicFeaturizer"),
    "FeatureEvaluator": ("FeatureEvaluator", "FeatureEvaluator")
}


def __getattr__(name):
    """Return a lazy proxy for heavy objects to avoid importing native extensions
    (scipy/sklearn) on mere package import. The real import happens when the
    returned proxy is called or an attribute accessed.
    """
    if name in _LAZY_MAP:
        module_subpath, attr_name = _LAZY_MAP[name]
        return _LazyImport(module_subpath, attr_name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

# --------------------------
# Support IDE autocomplete
# --------------------------
def __dir__():
    return sorted(__all__)
