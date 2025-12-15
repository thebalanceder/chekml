# Always available
from .IF.slow.inequality_based_featurization import (
    InequalityFeaturizer as InequalityFeaturizerSlow
)
from .IRF.slow.information_repurposed_featurization import (
    InformationRepurposedFeaturizer as InformationRepurposedFeaturizerSlow
)
__all__ = [
    "InequalityFeaturizerSlow",
    "InformationRepurposedFeaturizerSlow",
]

# Optional fast imports
try:
    from .IF.fast.inequality_based_featurization import (
        InequalityFeaturizer as InequalityFeaturizerFast
    )
    from .IRF.fast.information_repurposed_featurization import (
        InformationRepurposedFeaturizer as InformationRepurposedFeaturizerFast
    )
    __all__ += [
        "InequalityFeaturizerFast",
        "InformationRepurposedFeaturizerFast",
    ]
except Exception:
    # Fast modules silently disabled
    pass

