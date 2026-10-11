"""
EXPLAINA AI: Models Package
Exposes core model architectures, inference engines, and multimodal orchestrators.
"""

from .image_model import (
    build_mobilenet_v2_classifier,
    MalariaVisionModel,
    train_image_model,
    PatientCellDataset,
    build_patient_level_splits
)

from .clinical_model import (
    MalariaClinicalModel,
    train_clinical_models,
    CLINICAL_FEATURES,
    WHO_SEVERE_FEATURES,
    SYSTEMIC_FEATURES,
    evaluate_clinical_predictions
)

from .multimodal_orchestrator import (
    MultimodalOrchestrator,
    MULTIMODAL_METRICS
)

__all__ = [
    'build_mobilenet_v2_classifier',
    'MalariaVisionModel',
    'train_image_model',
    'PatientCellDataset',
    'build_patient_level_splits',
    'MalariaClinicalModel',
    'train_clinical_models',
    'CLINICAL_FEATURES',
    'WHO_SEVERE_FEATURES',
    'SYSTEMIC_FEATURES',
    'evaluate_clinical_predictions',
    'MultimodalOrchestrator',
    'MULTIMODAL_METRICS'
]
