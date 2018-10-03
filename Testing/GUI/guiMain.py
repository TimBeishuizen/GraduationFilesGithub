from PyQt5 import QtWidgets as QtW, QtCore as QtC, QtGui as QtG
import GUI.guiLoadDataset as loadingWidget

class MainFrame(QtW.QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Framework GUI'
        self.left = 10
        self.top = 10
        self.width = 540
        self.height = 380
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.form_widget = Framework()
        self.setCentralWidget(self.form_widget)

        self.init_menubar()


    def init_menubar(self):
        mainMenu = self.menuBar()
        datasetMenu = mainMenu.addMenu('Dataset')
        explorationMenu = mainMenu.addMenu('Exploration')
        preprocessingMenu = mainMenu.addMenu('Preprocessing')
        analysisMenu = mainMenu.addMenu('Analysis')

        LoadButton = QtW.QAction(QtG.QIcon('exit24.png'), 'Load', self)
        LoadButton.setShortcut('Ctrl+L')
        LoadButton.setStatusTip('Load a dataset')
        LoadButton.triggered.connect(self.load_dataset)

        SaveButton = QtW.QAction(QtG.QIcon('exit24.png'), 'Save', self)

        datasetMenu.addAction(LoadButton)
        datasetMenu.addAction(SaveButton)

        FeatureSelectionButton = QtW.QAction(QtG.QIcon('exit24.png'), 'Feature Selection', self)
        MissingValuesButton = QtW.QAction(QtG.QIcon('exit24.png'), 'Feature Selection', self)

        preprocessingMenu.addAction(FeatureSelectionButton)
        preprocessingMenu.addAction(MissingValuesButton)

        self.show()

    def load_dataset(self):
        self.load_widget = loadingWidget.OpenDataset()
        self.setCentralWidget(self.load_widget)


class Framework(QtW.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        button = QtW.QPushButton('PyQt5 button', self)
        button.setToolTip('This is an example button')
        button.move(100, 70)
        button.clicked.connect(self.on_click)

        # Create textbox
        self.textbox = QtW.QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280,40)

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
