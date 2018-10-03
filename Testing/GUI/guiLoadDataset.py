from PyQt5 import QtWidgets as QtW, QtCore as QtC, QtGui as QtG

class OpenDataset(QtW.QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.loadTitle = QtW.Q

        # Create textbox
        self.textbox = QtW.QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280, 40)

        button = QtW.QPushButton('Load .csv file', self)
        button.setToolTip('This is an example button')
        button.move(100, 70)
        button.clicked.connect(self.on_click)


    @QtC.pyqtSlot()
    def on_click(self):
        buttonReply = QtW.QMessageBox.question(self, 'PyQt5 message', "Do you want to write text?",
                                               QtW.QMessageBox.Yes | QtW.QMessageBox.No, QtW.QMessageBox.No)
        if buttonReply == QtW.QMessageBox.Yes:
            textboxValue = self.textbox.text()
            QtW.QMessageBox.question(self, 'Message - pythonspot.com', "You typed: " + textboxValue, QtW.QMessageBox.Ok,
                                 QtW.QMessageBox.Ok)
            self.textbox.setText("")
        else:
            print('No clicked.')
