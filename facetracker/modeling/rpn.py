from mmdet.models.detectors import RPN
from mmdet.models import DETECTORS

@DETECTORS.register_module()
class FaceTrackerRPN(RPN):
    test_mode = False
    """Implementation of Region Proposal Network"""
    def forward(self, img, img_metas=None):
        if self.test_mode:
            x = self.extract_feat(img)
            return self.rpn_head.simple_test_rpn(x, img_metas)
            
        return super(FaceTrackerRPN, self).forward(img, img_metas)