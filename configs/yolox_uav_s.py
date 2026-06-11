"""YOLOX-S experiment config for Anti-UAV v4 Track 3.

Mirrors the EXP used in training and the Kaggle inference notebook
(anti-uav-v4-yolox-inference-640p.ipynb). For inference, only `depth`,
`width`, `num_classes`, and `test_size` are read; the training paths and
hyperparameters are inert but kept here verbatim for parity with training.
"""
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 1
        self.exp_name = "yolox_uav_s"
        # Training-only paths (left as in the training notebook; not used at inference).
        self.data_dir = "/kaggle/working/yolox_data"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.input_size = (512, 640)
        self.test_size = (512, 640)
        self.multiscale_range = 0
        self.mosaic_prob = 1.0
        self.mosaic_scale = (0.8, 1.2)
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.degrees = 0.0
        self.shear = 0.0
        self.translate = 0.1
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.max_epoch = 50
        self.no_aug_epochs = 10
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.ema = True
        self.eval_interval = 1
        self.print_interval = 50
        self.data_num_workers = 4
        self.output_dir = "/kaggle/working/YOLOX_outputs"
