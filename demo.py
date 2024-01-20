import os
from mmengine import Config
from mmengine.runner import Runner
import warnings
warnings.filterwarnings("ignore")
work_dir = 'work_dir/SegTrue_1.5e-5_160k_b8_0.75-1.5-crop512-512'
ckpt = 'best_mIoU_iter_116000.pth'

test_cfg = Config.fromfile(f'{work_dir}/Tr_base2.0.py')
test_cfg.resume = False
test_cfg.load_from = os.path.join(work_dir, ckpt)
test_cfg.visualizer.save_dir = test_cfg.work_dir = os.path.join(work_dir, 'test')

test_runner = Runner.from_cfg(test_cfg)
test_runner.test()