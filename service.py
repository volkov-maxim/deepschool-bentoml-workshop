import bentoml
import numpy as np


# Настройки образа (Python, системные пакеты, зависимости) заданы в bentofile.yaml
@bentoml.service(
    resources={
        "cpu": "2",
        "memory": "4Gi",
        # "gpu": 1
    },
    workers=1,
)
class PlantSegmentator:
    def __init__(
        self,
        ckpt_path: str = "./checkpoint.pth",
    ) -> None:
        """
        Initialize the plant segmentation service.

        Args:
            ckpt_path (str): Path to the model checkpoint file (.pth) used to
                initialize the segmentation pipeline. Defaults to "./checkpoint.pth".
        """
        import torch
        from src.model_class import SegmentModelPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = SegmentModelPipeline(ckpt_path=ckpt_path, device=device)

    @bentoml.api(batchable=False)
    def run_processing(self, image: np.ndarray) -> np.ndarray:
        """
        Run plant segmentation on a single input image.

        Args:
            image (np.ndarray): Input image array in HWC format (height, width, channels),
                typically RGB with dtype uint8.

        Returns:
            np.ndarray: Segmentation result as an array 1HW float type ) from 0 to 1
        """

        results = self.pipeline.predict(image)
        return results

    @bentoml.api
    def get_model_version(self) -> str:
        """Get the version identifier of the currently loaded model."""

        return self.pipeline.model_version
