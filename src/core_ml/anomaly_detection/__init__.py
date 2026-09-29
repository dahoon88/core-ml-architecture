from .from_scratch import AnomalyDetector
from .framework import AnomalyDetector as SklearnAnomalyDetector

__all__ = ["AnomalyDetector", "SklearnAnomalyDetector"]
