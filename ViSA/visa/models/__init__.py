from .configuration import ABSARoBERTaConfig, ABSABERTConfig

from .hier_roberta_ml import HierRoBERTaML



ViSA_CONFIG_ARCHIVE_MAP = {
    "hier_roberta_ml": ABSARoBERTaConfig,
}

ViSA_MODEL_ARCHIVE_MAP = {
    "hier_roberta_ml": HierRoBERTaML,
}


__all__ = ["ABSARoBERTaConfig", "ViSA_MODEL_ARCHIVE_MAP", "ViSA_CONFIG_ARCHIVE_MAP"]
