from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class DetectedObject:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str
    mask: Optional[np.ndarray] = None


@dataclass
class AnalysisResult:
    original_image: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None
    objects: List[DetectedObject] = field(default_factory=list)

    total_count: int = 0
    red_count: int = 0
    yellow_count: int = 0
    egg_count: int = 0

    status_message: str = "Готово"