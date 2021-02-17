# blinkblink
python写的电脑端眨眼检测和提醒工具，提醒你每分钟眨眼N次  
我的眼睛时常会感觉干涩，本来以为眼睛干涩是眨眼次数不足引起，所以写了这个工具提醒自己眨眼  
但使用后发现眨眼次数基本达到标准，换了墨水屏显示器后眼睛感觉好多了，然后也暂停了优化

## 安装
pip install pip --upgrade  
pip install -r requirements.txt  
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

## 运行示例
- 眨眼检测测试  
python blinkdetector.py --camera_id 0
- 眨眼监测  
python blinkgui.py  
当每分钟眨眼次数小于设定值时，屏幕会出现悬浮窗，显示当前眨眼次数  
当眨眼次数大于设定值时，悬浮窗消失

## 程序结构说明
- facedetector 人脸检测  
感谢 https://github.com/ouyanghuiyu/yolo-face-with-landmark
- facetracker 人脸关键点追踪  
caltech数据集训练，代码基于mmdet
- blink 眨眼判断  
Closed Eyes In The Wild (CEW)数据集训练，代码基于mmdet