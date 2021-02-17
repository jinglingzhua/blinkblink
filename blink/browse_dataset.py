from task.common.browse_dataset import DBrowserBase
from modeling import *
from MyUtils.my_utils import *

class Browser(DBrowserBase):
    def show_item(self, item):
        show = item['img']
        cvshow(str(item['gt_labels']), show)

    
if __name__ == '__main__':
    Browser().run()