from GUI.guiMain import MainFrame
from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)
ex = MainFrame()
sys.exit(app.exec_())