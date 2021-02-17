from task.common.browse_dataset import *
from modeling import *
from MyUtils.my_utils import *

class Browser(DBrowserBase):
    def show_item(self, item):
        show = item['img']
        pts = item['gt_pts'].reshape((-1,2))
        print(pts)
        for x, y in np.int32(pts+0.5):
            cv2.circle(show, (x,y), 2, (0,255,0), thickness=2)
        cvshow(str(item['gt_labels']), show)
        
    def run(self):
        args = self._parser_args()
        cfg = Config.fromfile(args.config)
        data_cfg = eval(f'cfg.data.{args.data_type}')
        data_cfg['pipeline'] = [
            x for x in data_cfg.pipeline if x['type'] not in args.skip_type
        ]
    
        dataset = build_dataset(data_cfg)
    
        progress_bar = mmcv.ProgressBar(len(dataset))
        idx = list(range(len(dataset)))
        for i in idx:
            self.show_item(dataset[i])
            progress_bar.update()


    
if __name__ == '__main__':
    Browser().run()