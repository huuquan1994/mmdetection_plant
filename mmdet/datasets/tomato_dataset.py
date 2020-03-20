from .voc import VOCDataset
from .registry import DATASETS


@DATASETS.register_module
class TomatoDataset(VOCDataset):

    CLASSES = ('Cercospora leaf mold', 'Powdery mildew', 'Leaf mold', 'Gray mold', 'Corynespora cassiicola')