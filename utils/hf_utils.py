import datasets
import torch
import torch.utils.data as data
import torchvision

class ImageDatasetWithLabels(data.Dataset):
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        transform: torchvision.transforms.Compose = None,
    ) -> None:
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = item["label"]
        return image, label
