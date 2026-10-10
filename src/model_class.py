import segmentation_models_pytorch as smp
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentModelPipeline:
    def __init__(self, ckpt_path: str, device: str):
        self.model_version = "0.2"
        self.device = device
        self.transforms = A.Compose(
            [
                A.Resize(512, 512),
                A.Normalize(normalization="min_max"),
                ToTensorV2(),
            ]
        )

        self.model = smp.Unet(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            pooling="max",  # one of 'avg', 'max'
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            activation="sigmoid",  # the last activation function
        )
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def predict(self, image: np.ndarray) -> np.ndarray:
        preproc = self.transforms(image=image.astype(np.float32))
        image = preproc["image"].to(self.device)

        with torch.no_grad():
            pred = self.model(image.unsqueeze(0))
            pred = pred.cpu().numpy().squeeze(0)
        return pred
