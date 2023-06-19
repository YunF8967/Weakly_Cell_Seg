import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator

from Eval.DiceEvalHook import DiceEvalHook

def build_evaluator():
    pass

def buid_hook():
    pass

class MyTrainer(DefaultTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        # self.register_hooks([DiceEvalHook(), ])
    
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.RandomFlip(horizontal=True, vertical=False),
            T.RandomFlip(horizontal=False, vertical=True),
            T.RandomRotation(angle=[90, 180, 270])
            ])
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, ("segm",), output_dir = cfg.OUTPUT_DIR)