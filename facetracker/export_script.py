import torch
import numpy as np
from mmcv.runner import load_checkpoint
from task.common.fuse_conv_bn import fuse_module
from MyUtils.my_utils import *
from mmcv import Config
from modeling import *
from mmdet.models import build_detector  

def export(config_file, ckpt_file, dst):
    cfg = Config.fromfile(config_file)
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.eval()
    model.test_mode = True
    x = model.with_bbox
    
    checkpoint = load_checkpoint(model, ckpt_file, map_location='cpu')
    model = fuse_module(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    I = np.float32(np.random.rand(cfg.inp_hw[0],cfg.inp_hw[1],3))
    inp = torch.from_numpy(np.moveaxis(I,2,0))[None]  
    inp = inp.to(next(model.parameters()).device)
    traced_script_module = torch.jit.trace(model, inp, check_trace=False)
    with torch.no_grad():
        out_1 = traced_script_module(inp)
        out_2 = model(inp)
    for x, y in zip(out_1, out_2):
        assert np.ma.allclose(x.cpu().numpy(), y.cpu().numpy(), rtol=1e-3)
    traced_script_module.save(dst)
    print('export script done')

        
if __name__ == '__main__':
    from MyUtils.my_utils import opj
    model_dir = '/home/zmf/nfs/workspace/blink/model/facetracker_1010'    
    export(opj(model_dir, 'base.py'), opj(model_dir, 'latest.pth'),
           opj(model_dir, 'model.script'))