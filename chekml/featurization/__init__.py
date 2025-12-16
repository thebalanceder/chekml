# chekml/featurization/__init__.py

__all__ = [
    "InequalityFeaturizerSlow",
    "InformationRepurposedFeaturizerSlow",
    "InequalityFeaturizerFast",
    "InformationRepurposedFeaturizerFast",
]

# --------------------------
# Lazy attribute loader
# --------------------------
def __getattr__(name):
    """
    Lazily import heavy classes to avoid Kaggle kernel crashes.
    Called only when the attribute is accessed.
    """
    if name == "InequalityFeaturizerSlow":
        from .IF.slow.inequality_based_featurization import InequalityFeaturizer
        return InequalityFeaturizer

    if name == "InformationRepurposedFeaturizerSlow":
        from .IRF.slow.information_repurposed_featurization import InformationRepurposedFeaturizer
        return InformationRepurposedFeaturizer

    if name == "InequalityFeaturizerFast":
        try:
            from .IF.fast.inequality_based_featurization import InequalityFeaturizer
            return InequalityFeaturizer
        except Exception:
            raise ImportError("InequalityFeaturizerFast is not available in this environment.")

    if name == "InformationRepurposedFeaturizerFast":
        try:
            from .IRF.fast.information_repurposed_featurization import InformationRepurposedFeaturizer
            return InformationRepurposedFeaturizer
        except Exception:
            raise ImportError("InformationRepurposedFeaturizerFast is not available in this environment.")

    raise AttributeError(f"module {__name__} has no attribute {name}")

# --------------------------
# Support IDE autocomplete
# --------------------------
def __dir__():
    return sorted(__all__)
