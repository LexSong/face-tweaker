from pathlib import Path
from typing import Iterator
from typing import List
from typing import MutableMapping

from PIL import Image  # type: ignore


class ImageFolder(MutableMapping[str, Image.Image]):
    def __init__(
        self,
        folder,
        ext: str = ".jpg",
        image_save_params: dict = {"quality": 90},
    ):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.ext = ext
        self.image_save_params = image_save_params

    def get_image_path(self, name: str) -> Path:
        return self.folder / (name + self.ext)

    def get_image_names(self) -> List[str]:
        return sorted(x.stem for x in self.folder.iterdir() if x.suffix == self.ext)

    def __getitem__(self, name: str) -> Image.Image:
        image_path = self.get_image_path(name)
        if image_path.exists():
            return Image.open(image_path)
        else:
            raise KeyError(name)

    def __setitem__(self, name: str, image: Image.Image) -> None:
        image_path = self.get_image_path(name)
        image.save(image_path, **self.image_save_params)

    def __delitem__(self, name: str) -> None:
        image_path = self.get_image_path(name)
        if image_path.exists():
            image_path.unlink()
        else:
            raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.get_image_names())

    def __len__(self) -> int:
        return len(self.get_image_names())
