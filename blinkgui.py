import sys
from blinkstatus import BlinkStatus

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtWidgets import QLabel    

class BlinkGui(QLabel):

    def __init__(self, parent = None):
        QLabel.__init__(self, parent)

        self.init_gui()
        self.init_blink()
        
    def init_blink(self):
        self.bks = BlinkStatus()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.proc_blink)
        self.timer.start(33)
        
    def proc_blink(self):
        self.bks.proc()
        if self.bks.status == self.bks.DANGER:
            self.setText(str(self.bks.blinks_per_minute))
            self.setVisible(True)
        else:
            self.setVisible(False)
        
    def init_gui(self):
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint |\
                            Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setMouseTracking(True)

        sz = QSize(128,128)
        self.resize(sz)
        self.setMinimumSize(sz)
        self.setMaximumSize(sz)

        font = self.font()
        font.setPointSize(36)
        font.setBold(True)
        self.setFont(font)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mpos = event.globalPos() - self.pos()  
        if event.button() == Qt.RightButton:
            self.bks.reset()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.mpos)
            self.update()
        super().mouseMoveEvent(event)
        
    def enterEvent(self, event):
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.repaint()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication    
    app = QApplication(sys.argv)
    w = BlinkGui()
    w.show()
    sys.exit(app.exec_())
