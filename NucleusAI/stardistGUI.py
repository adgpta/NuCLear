from cProfile import run
from ctypes import alignment
from statistics import mode
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDesktopWidget, QMessageBox, QDialog, QApplication, QFileDialog, QMainWindow, QVBoxLayout, QListWidgetItem, QSlider, QStyleFactory, QSizePolicy
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QTransform, QIcon, QIntValidator
from PyQt5.QtCore import QUrl, QObject, pyqtSignal, QSize
from PyQt5 import QtCore
from PyQt5.uic import loadUi
from tensorboard import program
import os
import json
import glob
from utils import run_local
from threading import Thread
import platform
import webbrowser
import qimage2ndarray
from mask_overlay_func import overlay
import pyqtgraph as pg
import matplotlib.pyplot
import csv
import numpy as np
from tqdm import trange
from superqt import QRangeSlider
from qtpy.QtCore import Qt as Qtp
#import qdarkstyle

matplotlib.pyplot.switch_backend('Agg')

class MyException(Exception):
    pass

class Logger(QObject):
    finished = pyqtSignal(str)

class Resizer(QtCore.QObject):
    sizeChanged = QtCore.pyqtSignal(QtCore.QSize)

    def __init__(self, widget):
        super(Resizer, self).__init__(widget)
        self._widget = widget
        self.widget.installEventFilter(self)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, obj, event):
        if self.widget is obj and event.type() == QtCore.QEvent.Resize:
            self.sizeChanged.emit(event.size())
        return super(Resizer, self).eventFilter(obj, event)

class MainWindow(QMainWindow):
    factor = 1.5
    singleton: 'MainWindow' = None
    def __init__(self, parent=None):
        super(MainWindow,self).__init__(parent)
        loadUi("stardist.ui",self)
        #4ade00: Green,  #009deb:Blue,   #f95300:Orange
        StyleSheet = """
            Window{background: #b8cdee;}
            QToolBox::tab {
                background: #b6b7b8;
                border-radius: 1px;
                color: black;
                font-weight: bold;
            }
            QToolBox::tab:selected { /* italicize selected tabs */
                background: #404142;
                font: italic;
                font-weight: bold;
                color: white;
            }
            """
            
        self.toolBox_2.setStyleSheet(StyleSheet)
        #self.toolBox_2.itemText(0).setBold()
        print('--->>>>toolBox_2 :::: ',self.toolBox_2.itemText(0))

        #---------------------Saturation Test Starts-----------------------
        self.win = pg.GraphicsView()
        self.verticalLayout_3.addWidget(self.win)
        
        #----------------------frame number manually start--------------------------
        
        self.qlabel = QtWidgets.QLabel(text='Frame')
        self.frameno = QtWidgets.QLineEdit()
        self.onlyInt = QIntValidator()
        self.onlyInt.setBottom(0)
        self.frameno.setValidator(self.onlyInt)
        self.qlabel2 = QtWidgets.QLabel(text='/ -')
        self.pushjumpframe = QtWidgets.QPushButton('Go')

        frame1 = QtWidgets.QFrame()
        frame1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        #frame1.setStyleSheet("background-color: rgb(206, 210, 210)")
        frame1.setMaximumHeight(110)
        frame1.setMaximumWidth(210)
        #frame1.resize(100,100)
        
        self.qlabel.setFixedWidth(40)
        self.frameno.setFixedWidth(40)
        self.qlabel2.setFixedWidth(40)
        self.pushjumpframe.setFixedWidth(50)

        sublayout = QtWidgets.QGridLayout(frame1)
        sublayout.addWidget(self.qlabel,0,0)
        sublayout.addWidget(self.frameno,0,1)
        sublayout.addWidget(self.qlabel2,0,2)
        sublayout.addWidget(self.pushjumpframe,0,3)
        sublayout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_10.addLayout(sublayout, 0)
        self.verticalLayout_10.addWidget(frame1)
        
        self.pushjumpframe.setEnabled(False)
        self.pushjumpframe.clicked.connect(self.jumpFrame)
        #----------------------frame number manually ends--------------------------
        #.setGeometry(QRect(25, 71, 721, 480))

        self.slider = QRangeSlider(Qtp.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue([0, 255])
        #self.slider.setHigh(255)
        self.slider.setTickPosition(QSlider.TicksRight)
        self.slider.valueChanged.connect(self.level_change)
        self.gridLayout_24.addWidget(self.slider)

        self.NZ, self.Ly, self.Lx = 1,512,512
        self.saturation = [[0,255] for n in range(self.NZ)]

        #---------------------Saturation Test Ends-----------------------
        self.toolBox_2.setCurrentIndex(0)
        
        self.actionGitHub.triggered.connect(lambda: self.browser('stardistapp'))
        self.actionStardist.triggered.connect(lambda: self.browser('stardist'))
        self.actionPyradiomics.triggered.connect(lambda: self.browser('pyradiomics'))
        self.actionReport_Issue.triggered.connect(lambda: self.browser('issuepage'))

        self.pushjson.setEnabled(False)
        self.pushtboard.setEnabled(False)
        self.pushpredict.setEnabled(False)
        self.pushpredict_2.setEnabled(False)
        self.pushvalidate_5.setEnabled(False)

        self.inputbrowse.clicked.connect(self.ipbrowsefiles)
        self.outputbrowse.clicked.connect(self.opbrowsefiles)

        self.inputbrowse_2.clicked.connect(self.prediction_page_ip)
        self.outputbrowse_2.clicked.connect(self.prediction_page_op)
        self.modelbrowse_2.clicked.connect(self.prediction_page_modeldir) #change function of model dir

        self.inputbrowse_5.clicked.connect(self.validation_page_ip)
        self.outputbrowse_5.clicked.connect(self.validation_page_op)
        self.modelbrowse_5.clicked.connect(self.validation_page_modeldir)

        self.inputbrowse_4.clicked.connect(self.extraction_page_ip)
        self.outputbrowse_3.clicked.connect(self.extraction_page_op)

        #self.tboard.clicked.connect(self.tensorboard_push)
        self.train_parameters.clicked.connect(self.training_param)
        self.save_para = False
        self.pred_parameters.clicked.connect(self.prediction_param)
        self.pred_save_para = False
        self.val_parameters.clicked.connect(self.validation_param)
        self.val_save_para = False
        self.pushjson.clicked.connect(self.json_save)

        #extraction parameters settings
        self.extract_param_push.clicked.connect(self.extraction_param)
        self.extr_save_para = False
        
        self.pushtrain.clicked.connect(self.training_thread)
        self.pushvalidate_5.clicked.connect(self.validation_thread)
        self.pushpredict.clicked.connect(self.predict_thread)
        self.pushextract.clicked.connect(self.feature_extraction_thread)
        self.pushpredict_2.clicked.connect(self.predict_with_trained_model_thread)
        
        self.pushtboard.clicked.connect(self.tensorboard_new)
        
        self.validation_textbox.setReadOnly(True)
        self.validation_textbox.setUndoRedoEnabled(False)

        self.pushval_csv.setEnabled(False)
        self.pushval_csv.clicked.connect(self.save_validation_result)

        self.tboard_click = False

        self.pushtrain.setEnabled(False)
        self.pushtrain.setCheckable(True)

        self.Resetbutton.clicked.connect(MainWindow.restart)
        self.actionReset.triggered.connect(MainWindow.restart)

        self.image = None

        #--------------------------Frames-3D-----------------------
        self.prevFrameButton.setEnabled(False)
        self.nextFrameButton.setEnabled(False)
        self.prevFrameButton.clicked.connect(self.prevFrame)
        self.nextFrameButton.clicked.connect(self.nextFrame)

        self.previmg.setEnabled(False)
        self.nextimg.setEnabled(False)

        self.outlinesoncheck.toggled.connect(self.mask_outline_toggle_viewbox)
        self.maskoncheck.toggled.connect(self.mask_outline_toggle_viewbox)

        #-------------------------REVERT IMAGE TO ORIGINAL-------------------------
        self.reset_image_push.setEnabled(False)
        self.reset_image_push.clicked.connect(self.revertToOriginal)



        self.image_path = None
        self.NZ = 0
        self.image_frame = None
        self.ious_values = None
        self.supported_extensions = ('*.tif', '*.tiff')

        self.mask_loaded = False
        self.image_loaded = False

        self.l = Logger()
        self.counter_threads = 0

        if platform.system() == 'Windows':
            self.slash = '\\'
        else:
            self.slash = '/'

    def browser(self, action):
        if action == 'stardistapp':
            webbrowser.open('https://github.com/SFB1158RDM/Imaging_tools')
        elif action == 'stardist':
            webbrowser.open('https://github.com/stardist/stardist')
        elif action == 'pyradiomics':
            webbrowser.open('https://pyradiomics.readthedocs.io/')
        elif action == 'issuepage':
            webbrowser.open('https://github.com/SFB1158RDM/Imaging_tools/issues')
        pass

    def compute_saturation(self):
        # compute percentiles from stack
        self.saturation = []
        print('GUI_INFO: auto-adjust enabled, computing saturation levels')
        if self.NZ>10:
            iterator = trange(self.NZ)
        else:
            iterator = range(self.NZ)
        for n in iterator:
            self.saturation.append([np.percentile(self.stack[n].astype(np.float32),1),
                                    np.percentile(self.stack[n].astype(np.float32),99)])

    def level_change(self):
        sval = self.slider.value()
        self.ops_plot = {'saturation': sval}
        self.saturation[self.currentZ] = sval
        #if not self.autobtn.isChecked():
        for i in range(len(self.saturation)):
            self.saturation[i] = sval
        self.update_plot()

    def update_plot(self):        
        self.img.setLevels(self.saturation[self.currentZ])
        print('self.saturation[self.currentZ] : ', self.saturation[self.currentZ])


    @staticmethod
    def restart():
        #print('RESTART METHOD - 1')
        msgbox = QMessageBox(QMessageBox.Question, "Confirm clear", "Do you want to clear all settings?")
        msgbox.addButton(QMessageBox.Yes)
        msgbox.addButton(QMessageBox.No)
        msgbox.setDefaultButton(QMessageBox.No)
        
        reply = msgbox.exec()
        #print(reply)
    
        if reply == QMessageBox.Yes:
            print('Called Restart Method')
            os.execl(sys.executable, sys.executable, *sys.argv)
            
        else:
            return

    def training_thread(self):

        try:
            if self.inputfile != '' and self.opfname != '':
                self.pushtrain.setEnabled(False)
                self.pushtboard.setEnabled(True)
                self.submit.setEnabled(False)

                #i = 0
            
                self.t1=Thread(target=self.training, args=(self.counter_threads, self.l))
                self.t1.start()

                self.counter_threads += 1
                self.l.finished.connect(self.finished_thread)

                self.submit.setEnabled(True)
                #self.pushtrain.setEnabled(True)
            
        except AttributeError:
            msg8 = QtWidgets.QMessageBox()
            msg8.setIcon(QtWidgets.QMessageBox.Information)
            msg8.setText("Check if all the directories are set.")
            msg8.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg8.exec_()
            pass

    def finished_thread(self, message):
        print("finished: "+ message)
        self.counter_threads -= 1
        if self.counter_threads == 0:
            print("finished all threads")
            msg7 = QtWidgets.QMessageBox()
            msg7.setIcon(QtWidgets.QMessageBox.Information)
            msg7.setText("Process Completed.")
            msg7.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg7.exec_()
    

    def ipbrowsefiles(self):
        self.page_index = 'training_page'
        self.inputfile = QFileDialog.getExistingDirectory(self, 'Select training images folder', '')

        if not self.inputfile:
            return
        
        self._filenames = []
        self._masknames = []

        if platform.system() == 'Windows':
            self.inputfile = os.path.normpath(self.inputfile)
            
        
        self.path_to_image = os.path.join(self.inputfile, 'images')
        try:
            self.path_to_masks = os.path.join(self.inputfile, 'masks')
        except:
            self.path_to_masks = None
    
        for ext in self.supported_extensions:
            self._filenames.extend(glob.glob(self.path_to_image + self.slash + ext))
            try:
                self._masknames.extend(glob.glob(self.path_to_masks + self.slash + ext))
            except:
                pass
        
        print('self.path_to_images', self.path_to_image)
        if self.path_to_masks != None:
            print('self.path_to_masks', self.path_to_masks)
            assert len(self._filenames) == len(self._masknames), "Number of images and masks does not match"
            self.mask_loaded = True

        if self._filenames:
            self.image_loaded = True
            #show input file names in a list (list_filename)
            self.list_filename.clear()
            self.list_filename.addItems(self._filenames)

            self.inputdirtext.setText(self.path_to_image)
            self._current_index = 0
            self.display_images(self._filenames[self.current_index])

            self.filenm = self._filenames[self.current_index]
            self.filenm = self.filenm.rsplit(self.slash, 1)[1]
            
            if len(self._filenames)>1:
                self.nextimg.setEnabled(True)

            self.nextimg.clicked.connect(self.handle_next)
            self.previmg.clicked.connect(self.handle_previous)
            
        else:
            return

        self.list_filename.itemClicked.connect(self.display_images)
#--------------------START--next--previous--images-----------------------------------------

    def handle_next(self):
        if self.page_index == 'pred_page':
            self.current_index += 1
            self.display_images(self.pred_filenames[self.current_index])

            self.pred_filenm = self.pred_filenames[self.current_index]

            if platform.system() == 'Windows':
                self.pred_filenm = self.pred_filenm.rsplit('\\', 1)[1]
            else:
                self.pred_filenm = self.pred_filenm.rsplit('/', 1)[1]

            #self.imagelabel.setText('File: ' + self.pred_filenm)
           
        elif self.page_index == 'extraction_page':
            self.current_index += 1
            self.display_images(self.extract_filenames[self._current_index])
            
            self.extract_filenm = self.extract_filenames[self._current_index]

            if platform.system() == "Windows":
                self.extract_filenm = self.extract_filenm.rsplit('\\', 1)[1]
            else:
                self.extract_filenm = self.extract_filenm.rsplit('/', 1)[1]

            #self.imagelabel.setText('File: ' + self.extract_filenm)

        elif self.page_index == 'validation_page':
            self.current_index += 1
            self.display_images(self.val_filenames[self._current_index])
            
            self.val_filenm = self.val_filenames[self._current_index]

            if platform.system() == "Windows":
                self.val_filenm = self.val_filenm.rsplit('\\', 1)[1]
            else:
                self.val_filenm = self.val_filenm.rsplit('/', 1)[1]

        else:
            self.current_index += 1
            self.display_images(self._filenames[self.current_index])

            self.filenm = self._filenames[self.current_index]
            if platform.system() == 'Windows':
                self.filenm = self.filenm.rsplit('\\', 1)[1]
            else:
                self.filenm = self.filenm.rsplit('/', 1)[1]             

    def handle_previous(self):
        if self.page_index == 'pred_page':
            self.current_index -= 1
            self.display_images(self.pred_filenames[self.current_index])

            self.pred_filenm = self.pred_filenames[self.current_index]
            if platform.system() == 'Windows':
                self.pred_filenm = self.pred_filename.rsplit('\\', 1)[1]
            else:
                self.pred_filenm = self.pred_filename.rsplit('/', 1)[1]

        elif self.page_index == 'extraction_page':
            self.current_index -= 1
            self.display_images(self.extract_filenames[self._current_index])
            
            self.extract_filenm = self.extract_filenames[self._current_index]

            if platform.system() == "Windows":
                self.extract_filenm = self.extract_filenm.rsplit('\\', 1)[1]
            else:
                self.extract_filenm = self.extract_filenm.rsplit('/', 1)[1]

        elif self.page_index == 'validation_page':
            self.current_index -= 1
            self.display_images(self.val_filenames[self._current_index])
            
            self.val_filenm = self.val_filenames[self._current_index]

            if platform.system() == "Windows":
                self.val_filenm = self.val_filenm.rsplit('\\', 1)[1]
            else:
                self.val_filenm = self.val_filenm.rsplit('/', 1)[1]

        else:
            self.current_index -= 1
            self.display_images(self._filenames[self.current_index])

            self.filenm = self._filenames[self.current_index]
            if platform.system() == 'Windows':
                self.filenm = self.filenm.rsplit('\\', 1)[1]
            else:
                self.filenm = self.filenm.rsplit('/', 1)[1]

    @property
    def current_index(self):
        return self._current_index

    @current_index.setter
    def current_index(self, index):
        if self.page_index == 'pred_page':
            if index <= 0:
                self._update_button_status(False, True)
            elif index >= (len(self.pred_filenames) - 1):
                self._update_button_status(True, False)
            else:
                self._update_button_status(True, True)

            if 0 <= index < len(self.pred_filenames):
                self._current_index = index
                filename = self.pred_filenames[self._current_index]

        elif self.page_index == 'extraction_page':
            if index <= 0:
                self._update_button_status(False, True)
            elif index >= (len(self.extract_filenames) - 1):
                self._update_button_status(True, False)
            else:
                self._update_button_status(True, True)

            if 0 <= index < len(self.extract_filenames):
                self._current_index = index
                filename = self.extract_filenames[self._current_index]

        elif self.page_index == 'validation_page':
            print('VALIDATE INDEX IS = ', index)
            if index <= 0:
                self._update_button_status(False, True)
            elif index >= (len(self.val_filenames) - 1):
                self._update_button_status(True, False)
            else:
                self._update_button_status(True, True)

            if 0 <= index < len(self.val_filenames):
                self._current_index = index
                filename = self.val_filenames[self._current_index]

        else:
            print('TRAIN INDEX IS = ', index)    
            if index <= 0:
                self._update_button_status(False, True)
            elif index >= (len(self._filenames) - 1):
                self._update_button_status(True, False)
            else:
                self._update_button_status(True, True)

            if 0 <= index < len(self._filenames):
                self._current_index = index
                filename = self._filenames[self._current_index]
           
            #self.imagelabel.setText('File: ' + filename)

    def _update_button_status(self, previous_enable, next_enable):
        self.previmg.setEnabled(previous_enable)
        self.nextimg.setEnabled(next_enable) 

#--------------------END--next--previous--images-----------------------------------------

    def opbrowsefiles(self):
        self.opfname=QFileDialog.getExistingDirectory(self, 'Open directory', '')

        if platform.system() == "Windows":
            self.opfname = os.path.normpath(self.opfname)

        if self.opfname:
            self.outputdirtext.setText(self.opfname)
        else:
            return

#--------------------Start prediction_page user inputs-----------------------------------------
    def prediction_page_ip(self):
        self.page_index = 'pred_page'
        self.pred_ipfilename = QFileDialog.getExistingDirectory(self, 'Open directory', '')

        if not self.pred_ipfilename:
            return

        self.pred_filenames = []

        if platform.system() == "Windows":
            self.pred_ipfilename = os.path.normpath(self.pred_ipfilename)

        self.pred_path_to_image = os.path.join(self.pred_ipfilename, 'images')
        
        for ext in self.supported_extensions:
            self.pred_filenames.extend(glob.glob(self.pred_path_to_image + self.slash + ext))

        print('self.path_to_images', self.pred_path_to_image)

        if self.pred_filenames:   
            self.image_loaded = True
            self.inputdirtext_2.setText(self.pred_ipfilename)  
            self._current_index = 0
            self.display_images(self.pred_filenames[self._current_index])
            
            self.pred_filenm = self.pred_filenames[self._current_index]
            
            self.pred_filenm = self.pred_filenm.rsplit(self.slash, 1)[1]
            
            self.nextimg.clicked.connect(self.handle_next)
            self.previmg.clicked.connect(self.handle_previous)
    
        

    def prediction_page_op(self):
        self.pred_opfilename = QFileDialog.getExistingDirectory(self, 'Open directory', '')

        if platform.system() == "Windows":
            self.pred_opfilename = os.path.normpath(self.pred_opfilename)

        if self.pred_opfilename:
            self.outputdirtext_2.setText(self.pred_opfilename)
            print('output dir is set.')
        else:
            return

    def prediction_page_modeldir(self):
        self.pred_modeldir = QFileDialog.getExistingDirectory(self, 'Open directory', '')

        if platform.system() == "Windows":
            self.pred_modeldir = os.path.normpath(self.pred_modeldir)

        if self.pred_modeldir:
            self.modeldir_2.setText(self.pred_modeldir)
            print('model dir is set.')
        else:
            return

#--------------------End prediction_page user inputs-----------------------------------------

#--------------------Start validation_page user inputs-----------------------------------------
    def validation_page_ip(self):
        self.page_index = 'validation_page'
        self.val_ipfilename = QFileDialog.getExistingDirectory(self, 'Open directory', '')

        if not self.val_ipfilename:
            return

        self.val_filenames = []
        self._masknames = []

        if platform.system() == "Windows":
            self.val_ipfilename = os.path.normpath(self.val_ipfilename)

        self.val_path_to_image = os.path.join(self.val_ipfilename, 'images')
        try:
            self.val_path_to_masks = os.path.join(self.val_ipfilename, 'masks')
        except:
            self.val_path_to_masks = None

        
        for ext in self.supported_extensions:
            self.val_filenames.extend(glob.glob(self.val_path_to_image + self.slash + ext))
            try:
                self._masknames.extend(glob.glob(self.val_path_to_masks + self.slash + ext))
            except:
                pass
        #print('NUMBER OF VALIDATION IMAGES:', len(self.val_filenames))
        print('self.path_to_validation_images:', self.val_path_to_image)
        if self.val_path_to_masks != None:
            print('self.path_to_masks', self.val_path_to_masks)
            assert len(self.val_filenames) == len(self._masknames), "Number of images and masks does not match"
            self.mask_loaded = True

        if self.val_filenames:   
            self.image_loaded = True
            self.inputdirtext_5.setText(self.val_ipfilename)  
            self._current_index = 0
            self.display_images(self.val_filenames[self._current_index])
            
            self.val_filenm = self.val_filenames[self._current_index]
            self.val_filenm = self.val_filenm.rsplit(self.slash, 1)[1]

            if len(self.val_filenames)>1:
                self.nextimg.setEnabled(True)
            
            self.nextimg.clicked.connect(self.handle_next)
            self.previmg.clicked.connect(self.handle_previous)

    def validation_page_op(self):
        self.val_opfilename = QFileDialog.getExistingDirectory(self, 'Open directory', '')

        if platform.system() == "Windows":
            self.val_opfilename = os.path.normpath(self.val_opfilename)

        if self.val_opfilename:
            self.outputdirtext_5.setText(self.val_opfilename)
            print('output dir is set.')
        else:
            return

    def validation_page_modeldir(self):
        self.val_modeldir = QFileDialog.getExistingDirectory(self, 'Open directory', '')

        if platform.system() == "Windows":
            self.val_modeldir = os.path.normpath(self.val_modeldir)

        if self.val_modeldir:
            self.modeldir_5.setText(self.val_modeldir)
            print('model dir is set.')
        else:
            return

#--------------------End validation_page user inputs-----------------------------------------

#--------------------Start extraction_page user inputs-----------------------------------------
    def extraction_page_ip(self):
        self.page_index = 'extraction_page'
        self.extract_ipfilename = QFileDialog.getExistingDirectory(self, 'Open directory', '')

        if platform.system() == "Windows":
            self.extract_ipfilename = os.path.normpath(self.extract_ipfilename)

        if self.extract_ipfilename:
            self.inputdirtext_4.setText(self.extract_ipfilename)
            print('input dir is set.')
        else:
            return

        self.extract_path_to_image = os.path.join(self.extract_ipfilename, 'images')
        try:
            self.extract_path_to_masks = os.path.join(self.extract_ipfilename, 'masks')
        except:
            self.extract_path_to_masks = None

        self.extract_filenames = []
        self._masknames = []
        
        for ext in self.supported_extensions:
            self.extract_filenames.extend(glob.glob(self.extract_path_to_image + self.slash + ext))
            try:
                self._masknames.extend(glob.glob(self.extract_path_to_masks + self.slash + ext))
            except:
                pass
        
        if self.extract_path_to_masks != None:
            print('self.path_to_masks', self.extract_path_to_masks)
            assert len(self.extract_filenames) == len(self._masknames), "Number of images and masks does not match"
            self.mask_loaded = True

        if self.extract_filenames:     
            self.image_loaded = True
            self._current_index = 0
            self.display_images(self.extract_filenames[self._current_index])
            
            self.extract_filenm = self.extract_filenames[self._current_index]

            if platform.system() == "Windows":
                self.extract_filenm = self.extract_filenm.rsplit('\\', 1)[1]
            else:
                self.extract_filenm = self.extract_filenm.rsplit('/', 1)[1]

            if len(self.extract_filenames)>1:
                self.nextimg.setEnabled(True)
            
            self.nextimg.clicked.connect(self.handle_next)
            self.previmg.clicked.connect(self.handle_previous)

    def extraction_page_op(self):
        self.extract_opfilename = QFileDialog.getExistingDirectory(self, 'Open directory', '')
        if platform.system() == "Windows":
            self.extract_opfilename = os.path.normpath(self.extract_opfilename)

        if self.extract_opfilename:
            self.outputdirtext_3.setText(self.extract_opfilename)
            print('output dir is set.')
        else:
            return

#--------------------End extraction_page user inputs-----------------------------------------

#--------------------Reset Image to Original-----------------------------------------

    def revertToOriginal(self):
        print("Revert Image")
        if self.image_path:
            self.display_images(self.image_path, 0)
        """Revert the image back to original image."""
        ##TO-DO: Display message dialohg to confirm actions
        print("Done Revert Image")

    def prevFrame(self):
        """ Show previous frame in stack."""
        self.display_images(self.image_path, self.currentZ-1)
    
    def nextFrame(self):
        """ Show next frame in stack."""
        self.display_images(self.image_path, self.currentZ+1)

    def jumpFrame(self):
        """ Show selected frame in stack."""
        if self.frameno.text() != '' and self.image_path and self.currentZ != (int(self.frameno.text()) - 1):
            if int(self.frameno.text()) <= self.NZ:
                self.display_images(self.image_path, int(self.frameno.text()) - 1)
        else:
            pass

    #CHECK
    def mask_outline_toggle(self): #NEW
        if self.page_index == 'training_page':
            if len(self._filenames)>0 and len(self._masknames)>0:
                self.display_images(self.image_path, self.currentZ)
        elif self.page_index == 'pred_page':
            if len(self.pred_filenames)>0 and len(self._masknames)>0:
                self.display_images(self.image_path, self.currentZ)

    def mask_outline_toggle_viewbox(self): #NEW
        if self.mask_loaded:
            if not self.maskoncheck.isChecked() and not self.outlinesoncheck.isChecked():
                #self.p0.removeItem(self.img2)
                #self.p0.
                self.img2.clear()
            else:
                if self.page_index == 'training_page':
                    if len(self._filenames)>0 and len(self._masknames)>0:
                        print('-------> toggled', self.maskoncheck.isChecked(), self.outlinesoncheck.isChecked())
                        self.display_images(self.image_path, self.currentZ)
                elif self.page_index == 'pred_page':
                    if len(self.pred_filenames)>0 and len(self._masknames)>0:
                        self.display_images(self.image_path, self.currentZ)
                elif self.page_index == 'extraction_page':
                    if len(self.extract_filenames)>0 and len(self._masknames)>0:
                        self.display_images(self.image_path, self.currentZ)
                elif self.page_index == 'validation_page':
                    if len(self.val_filenames)>0 and len(self._masknames)>0:
                        self.display_images(self.image_path, self.currentZ)
        else:
            pass

    def display_images(self, image_path, FrameIndex=0): #NEW
        
        if self.image_path != image_path:
            self.new_img = True
        else:
            self.new_img = False

        print('790',self.new_img)
        if self.new_img == True:
            print('794')
            self.image_path = image_path
            self.img_path_str = self.to_string(image_path)
            if platform.system() == 'Windows':
                self.filenm = self.img_path_str.rsplit('\\', 1)[1]
            else:
                self.filenm = self.img_path_str.rsplit('/', 1)[1]

            if isinstance(image_path, str):
                image_path = image_path
            elif isinstance(image_path, QListWidgetItem):
                image_path = image_path.text()
            else:
                print('TypeError: QPixmap(): argument 1 has unexpected type.')
            
            #check if image has stacks
            self.parent = overlay.imread(image_path)
            self.stack, self.NZ = overlay._initialize_image(self.parent)
            self.compute_saturation()
            

        #---------For 3D images with stack length more than 1. For 2D, NZ remains 1.
        if self.NZ > 0: # If image is not None. 
            self.currentZ = FrameIndex
            show_stack = True
            if self.NZ <= 1:
                self.prevFrameButton.setEnabled(False)
                self.nextFrameButton.setEnabled(False)
            else:
                if (self.currentZ<=0):
                    self.prevFrameButton.setEnabled(False)
                    self.nextFrameButton.setEnabled(True)
                elif (0 < self.currentZ < (self.NZ-1)):
                    self.prevFrameButton.setEnabled(True)
                    self.nextFrameButton.setEnabled(True)
                elif (self.currentZ==(self.NZ-1)): 
                    self.prevFrameButton.setEnabled(True)
                    self.nextFrameButton.setEnabled(False)

            self.image_frame = self.stack[self.currentZ,:,:,0]
            
        self.p0 = pg.ViewBox(border=[100, 100, 100], lockAspect=True, invertY=True, enableMouse=True)
        self.p0.setMenuEnabled(False)
        self.win.setCentralItem(self.p0)
        self.win.show()

        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.p0.addItem(self.img)

        self.img2 = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img2.autoDownsample = False
        self.p0.addItem(self.img2)

        self.img.setImage(self.image_frame, autoLevels=False)
        #self.img.setLevels([0.0, 255.0])
        self.img.setLevels(self.saturation[self.currentZ])

        self.slider.setValue([self.saturation[self.currentZ][0], self.saturation[self.currentZ][1]])

        if show_stack:
            self.imagelabel.setText('Image File: ' + self.filenm)
            self.frameno.setText(str(self.currentZ+1))
            self.qlabel2.setText("/ " + str(self.NZ))
            #self.framesliderlabel.setText('Frame: ' + str(self.currentZ+1) + "/" + str(self.NZ))
        
        if len(self._masknames)>0 and (self.maskoncheck.isChecked() or self.outlinesoncheck.isChecked()):
            masksOn = self.maskoncheck.isChecked()
            outlinesOn = self.outlinesoncheck.isChecked()
            self.maskfile = ' '.join([str(f) for f in self._masknames if self.filenm in f])

            #overlay.draw_layer returns an array
            self.mask_image = overlay.draw_layer(image_path=image_path, masks_path=self.maskfile, currentZ=self.currentZ, masksOn=masksOn, outlinesOn=outlinesOn)

            self.img2.setImage(self.mask_image , autoLevels=False)
            self.img2.setLevels([0.0, 255.0])

        elif not self.maskoncheck.isChecked() and not self.outlinesoncheck.isChecked():
            #self.p0.removeItem()
            self.img2.clear()
            pass
        
        #Whenever the image changes, values of brightness and contrast returns to its default.
        self.reset_image_push.setEnabled(True)
        self.pushjumpframe.setEnabled(True)
        
    def keyPressEvent(self, event): #NEW
        try:
            if self.NZ>1:
                if event.key() == QtCore.Qt.Key_Left and self.current_index>0:
                    self.prevFrame()
                elif event.key() == QtCore.Qt.Key_Right and self.current_index<self.NZ:
                    self.nextFrame()
                else:
                    pass
        except:
            pass

    def to_string(self, inp):
        if type(inp) != str:
            str1 = inp.text()
        else:
            str1 = inp

        return str1

    def tensorboard_logfile(self):
        print('tensorboard_logfile')
        if self.outputdirtext.text() != '':
            print(os.path.join(self.outputdirtext.text(), self.config['modelName'] , 'logs'))
            self.logspath = os.path.join(self.outputdirtext.text(), self.config['modelName'] , 'logs')
            
            if platform.system() == "Windows":
                self.logspath = os.path.normpath(self.logspath)
            return self.logspath
        else:
            return None

    def tensorboard_new(self):

        if not self.tboard_click:
            self.tb = program.TensorBoard()
            #pathtb = "path/to/logs"

            pathtb = self.tensorboard_logfile()
            if platform.system() == "Windows":
                self.tb.configure(logdir=os.path.expanduser(pathtb), host='127.0.0.1', port=8080)
            else:    
                self.tb.configure(logdir=os.path.expanduser(pathtb), host='0.0.0.0', port=8080)
            self.url = self.tb.launch()
            print(self.url)
            self.tboard_click = True

        vbox = QVBoxLayout()
        webEngineView = QWebEngineView()
        webEngineView.load(QUrl(self.url))

        vbox.addWidget(webEngineView)
        w_tb = QDialog(self)
        w_tb.setLayout(vbox)        
        w_tb.setMinimumSize(900,700)
        #self.setGeometry(300, 300, 350, 250)
        w_tb.setWindowTitle('Tensorboard QWebEngineView')
        w_tb.show()


    def json_save(self):
        self.config = dict()
        self.config['modelName'] = self.inputs['Model Name'].text()
        self.config['valFraction'] = self.inputs['Validation Fraction'].text()
        self.config['patchSizeH'] = self.inputs['Patch Size Height'].text()
        self.config['patchSizeW'] = self.inputs['Patch Size Width'].text()
        self.config['patchSizeD'] = self.inputs['Patch Size Depth (only when 3D)'].text()
        self.config['extension'] = self.inputs['Input File Format Extension'].text()

        self.config['twoDim'] = self.inputs['2D'].isChecked()
        self.config['saveForFiji'] = self.inputs['Save model for ImageJ/Fiji'].isChecked()
        self.config['multichannel'] = False

        if self.outputdirtext.text() != '':
            with open(self.outputdirtext.text() + self.slash + "parameters.json", "w") as outfile: 
                json.dump(self.config, outfile)
                msg3 = QtWidgets.QMessageBox()
                msg3.setIcon(QtWidgets.QMessageBox.Information)
                msg3.setText("Parameters are saved as JSON file.")
                msg3.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg3.exec_()
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Please specify output directory!")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()

    def close(self):
        print('parameters are set.')
        self.close()

#--------------------Start training_page parameters-----------------------------------------
    def save_training_parameters(self):
        self.save_para = True
        
        self.param = dict()

        self.param['modelname'] = self.inputs['Model Name']

        self.param['valFrac'] = self.inputs['Validation Fraction']
        self.param['PatchH'] = self.inputs['Patch Size Height']
        self.param['PatchW'] = self.inputs['Patch Size Width'] 
        self.param['PatchD'] = self.inputs['Patch Size Depth (only when 3D)']
        self.param['ext'] = self.inputs['Input File Format Extension']
        self.param['2D'] = self.inputs['2D']
        self.param['saveforFiji'] = self.inputs['Save model for ImageJ/Fiji']
        #self.param['multichannel'] = False
        self.param['epochs'] = self.inputs['Epochs']
        self.param['steps_per_epoch'] = self.inputs['Steps per Epoch']
        self.param['n_rays'] = self.inputs['N_rays (Recommended to use a power of 2.)']
        
        self.pushjson.setEnabled(True)

        print('Param ::::', '\n', self.param)
        print('type of Param ::: ', type(self.param))

        self.w.close()
        self.pushtrain.setEnabled(True)
        
    def training_param(self):

        self.w = QDialog(self)
        mainLayout = QtWidgets.QFormLayout(fieldGrowthPolicy=1)
        self.inputs = dict()

        if not self.save_para:
            self.inputs['Model Name'] = QtWidgets.QLineEdit(text='model_xxyyzz') 
            self.inputs['Model Name'].setToolTip('Enter name of the model.')
            self.inputs['Validation Fraction'] = QtWidgets.QLineEdit(text='.1')
            self.inputs['Validation Fraction'].setToolTip('The fraction of available data that is used for validation, default')
            self.inputs['Patch Size Height'] = QtWidgets.QLineEdit(text='128')
            self.inputs['Patch Size Height'].setToolTip('Size (Height) of the image patches used to train the network')
            self.inputs['Patch Size Width'] = QtWidgets.QLineEdit(text='128')
            self.inputs['Patch Size Width'].setToolTip('Size (Width) of the image patches used to train the network')
            self.inputs['Patch Size Depth (only when 3D)'] = QtWidgets.QLineEdit()
            self.inputs['Patch Size Depth (only when 3D)'].setToolTip('Size (Depth) of the image patches used to train the network, only with 3D data.')
            self.inputs['Input File Format Extension'] = QtWidgets.QLineEdit(text='.tif')
            self.inputs['Input File Format Extension'].setToolTip('Input images file extension.')
            self.inputs['Epochs'] = QtWidgets.QLineEdit(text='100')
            self.inputs['Epochs'].setToolTip('Number of trainig epochs')
            self.inputs['Steps per Epoch'] = QtWidgets.QLineEdit(text='400')
            self.inputs['Steps per Epoch'].setToolTip('Number of parameter update steps per epoch.')
            self.inputs['N_rays (Recommended to use a power of 2.)'] = QtWidgets.QLineEdit(text='32')
            self.inputs['N_rays (Recommended to use a power of 2.)'].setToolTip('Number of radial directions for the star-convex polygon. Recommended to use a power of 2')
            self.inputs['2D'] = QtWidgets.QCheckBox()
            self.inputs['2D'].setToolTip('Select to train 2D model')
            self.inputs['2D'].setChecked(True)
            self.inputs['Save model for ImageJ/Fiji'] = QtWidgets.QCheckBox()
            self.inputs['Save model for ImageJ/Fiji'].setToolTip('Save the model for FIJI')
            #self.inputs['Multichannel'] = QtWidgets.QCheckBox()
        else:
            self.inputs['Model Name'] = self.param['modelname']
            self.inputs['Validation Fraction'] = self.param['valFrac']
            self.inputs['Patch Size Height'] = self.param['PatchH']
            self.inputs['Patch Size Width'] = self.param['PatchW']
            self.inputs['Patch Size Depth (only when 3D)'] = self.param['PatchD']
            self.inputs['Input File Format Extension'] = self.param['ext']
            self.inputs['Epochs'] = self.param['epochs']
            self.inputs['Steps per Epoch'] = self.param['steps_per_epoch']
            self.inputs['N_rays (Recommended to use a power of 2.)'] = self.param['n_rays']
            self.inputs['2D'] = self.param['2D']
            self.inputs['Save model for ImageJ/Fiji'] = self.param['saveforFiji']
            #self.inputs['Multichannel'] = self.param['multichannel']

        for label, widget in self.inputs.items():
            mainLayout.addRow(label, widget)

        self.submit = QtWidgets.QPushButton('Set')
        #self.submit.clicked.connect(w.close)
        self.submit.clicked.connect(self.save_training_parameters)
        mainLayout.addRow(self.submit)
        self.w.setLayout(mainLayout)
        self.w.setMinimumSize(500, 400)
        self.w.setMaximumHeight(450)
        #self.w.setFixedSize(600, 350)
        self.w.setWindowTitle('Set Training Parameters')
        self.w.show()

#--------------------End training_page parameters------------------------------------------

#--------------------Start prediction_page parameters-----------------------------------------
    def save_prediction_parameters(self):
        self.pred_save_para = True
        self.pushpredict_2.setEnabled(True)

        self.pred_param = dict()

        self.pred_param['2D'] = self.pred_inputs['2D']
        #self.pred_param['Run_local'] = self.pred_inputs['Run Local']
        self.pred_param['ext'] = self.pred_inputs['Input File Format Extension']

        print('Prediction Parameters ::::', '\n', self.pred_param)
        self.pred_w.close()

    def prediction_param(self):

        self.pred_w = QDialog(self)
        pred_mainLayout = QtWidgets.QFormLayout(fieldGrowthPolicy=1)
        self.pred_inputs = dict()

        if not self.pred_save_para:
            #self.pred_inputs['Run Local'] = QtWidgets.QCheckBox()
            self.pred_inputs['Input File Format Extension'] = QtWidgets.QLineEdit(text='.tif')
            self.pred_inputs['Input File Format Extension'].setToolTip('Input images file extension.')
            self.pred_inputs['2D'] = QtWidgets.QCheckBox()
            self.pred_inputs['2D'].setToolTip('Select to predict with 2D model')
            self.pred_inputs['2D'].setChecked(True)
            
            #self.inputs['Multichannel'] = QtWidgets.QCheckBox()
        else:
            self.pred_inputs['Input File Format Extension'] = self.pred_param['ext']
            self.pred_inputs['2D'] = self.pred_param['2D']

        for label, widget in self.pred_inputs.items():
            pred_mainLayout.addRow(label, widget)

        self.pred_submit = QtWidgets.QPushButton('Set')
        #self.submit.clicked.connect(w.close)
        self.pred_submit.clicked.connect(self.save_prediction_parameters)
        pred_mainLayout.addRow(self.pred_submit)
        self.pred_w.setLayout(pred_mainLayout)
        self.pred_w.setMinimumSize(500, 400)
        self.pred_w.setMaximumHeight(450)
        #self.w.setFixedSize(600, 350)
        self.pred_w.setWindowTitle('Set Parameters for Prediction')
        self.pred_w.show()

#--------------------End prediction_page parameters-----------------------------------------

#--------------------Start validation_page parameters-----------------------------------------
    def save_validation_parameters(self):
        self.val_save_para = True
        self.pushvalidate_5.setEnabled(True)

        self.val_param = dict()
        self.val_param['2D'] = self.val_inputs['2D']
        self.val_param['ext'] = self.val_inputs['Input File Format Extension']

        print('Validation Parameters ::::', '\n', self.val_param['2D'].isChecked(),self.val_param['ext'].text())
        self.val_w.close()

    def validation_param(self):

        self.val_w = QDialog(self)
        val_mainLayout = QtWidgets.QFormLayout(fieldGrowthPolicy=1)
        self.val_inputs = dict()

        if not self.val_save_para:
            self.val_inputs['Input File Format Extension'] = QtWidgets.QLineEdit(text='.tif')
            self.val_inputs['Input File Format Extension'].setToolTip('Input images file extension.')
            self.val_inputs['2D'] = QtWidgets.QCheckBox()
            self.val_inputs['2D'].setToolTip('Select to predict with 2D model')
            self.val_inputs['2D'].setChecked(True)
            
            #self.inputs['Multichannel'] = QtWidgets.QCheckBox()
        else:
            self.val_inputs['Input File Format Extension'] = self.val_param['ext']
            self.val_inputs['2D'] = self.val_param['2D']

        for label, widget in self.val_inputs.items():
            val_mainLayout.addRow(label, widget)

        self.val_submit = QtWidgets.QPushButton('Set')
        #self.submit.clicked.connect(w.close)
        self.val_submit.clicked.connect(self.save_validation_parameters)
        val_mainLayout.addRow(self.val_submit)
        self.val_w.setLayout(val_mainLayout)
        self.val_w.setMinimumSize(500, 400)
        self.val_w.setMaximumHeight(450)
        #self.w.setFixedSize(600, 350)
        self.val_w.setWindowTitle('Set Parameters for Prediction')
        self.val_w.show()

#--------------------End validation_page parameters-----------------------------------------

#--------------------Start extraction_page parameters-----------------------------------------
    def save_extraction_parameters(self):
        self.extr_save_para = True

        self.extr_param = dict()

        self.extr_param['Number of threads'] = self.extr_inputs['Number of Threads']
        self.extr_param['2D'] = self.extr_inputs['2D']

        self.extr_w.close()

    def extraction_param(self):

        self.extr_w = QDialog(self)
        extr_mainLayout = QtWidgets.QFormLayout(fieldGrowthPolicy=1)
        self.extr_inputs = dict()

        if not self.extr_save_para:
            self.extr_inputs['Number of Threads'] = QtWidgets.QLineEdit(text='12')
            self.extr_inputs['2D'] = QtWidgets.QCheckBox()
            self.extr_inputs['2D'].setChecked(True)
        else:
            self.extr_inputs['Number of Threads'] = self.extr_param['Number of threads']
            self.extr_inputs['2D'] = self.extr_param['2D']

        for label, widget in self.extr_inputs.items():
            extr_mainLayout.addRow(label, widget)

        self.extr_submit = QtWidgets.QPushButton('Set')
        #self.submit.clicked.connect(w.close)
        self.extr_submit.clicked.connect(self.save_extraction_parameters)
        extr_mainLayout.addRow(self.extr_submit)
        self.extr_w.setLayout(extr_mainLayout)
        self.extr_w.setMinimumSize(500, 400)
        self.extr_w.setMaximumHeight(450)
        #self.w.setFixedSize(600, 350)
        self.extr_w.setWindowTitle('Set Parameters for Feature Extraction')
        self.extr_w.show()

#--------------------End extraction_page parameters-----------------------------------------

#--------------------Start training-----------------------------------------
        
    def training(self, thread_count, log):
        
        self.pushtboard.setEnabled(True)
        self.pushpredict.setEnabled(True)

        try:
            self.config = dict()

            self.config['inputDir'] = self.inputfile
            self.config['outputDir'] = self.opfname 

            self.config['modelName'] = self.inputs['Model Name'].text()
            self.config['valFraction'] = self.inputs['Validation Fraction'].text()
            self.config['patchSizeH'] = self.inputs['Patch Size Height'].text()
            self.config['patchSizeW'] = self.inputs['Patch Size Width'].text()
            self.config['patchSizeD'] = self.inputs['Patch Size Depth (only when 3D)'].text()
            self.config['extension'] = self.inputs['Input File Format Extension'].text()

            self.config['twoDim'] = self.inputs['2D'].isChecked()
            self.config['saveForFiji'] = self.inputs['Save model for ImageJ/Fiji'].isChecked()
            self.config['multichannel'] = False
            #self.config['runLocal'] = self.runlocal.isChecked()
            self.config['epochs'] = self.inputs['Epochs'].text()
            self.config['steps_per_epoch'] = self.inputs['Steps per Epoch'].text()
            self.config['n_rays'] = self.inputs['N_rays (Recommended to use a power of 2.)'].text()

            print('Configuration for run_local ::::', '\n', self.config)
            print('Type of Config', type(self.config))

        except:
            
            raise MyException("An error occured. Please check if all the inputs are given.")


        #if self.runlocal.isChecked():
        run_local(self.config, destination='training')
        log.finished.emit("thread %d" % thread_count)
#--------------------End training-----------------------------------------           

    def predict_thread(self):
        
        self.t3 = Thread(target=self.predict, args=(self.counter_threads, self.l))
        self.t3.start()

        self.counter_threads += 1
        self.l.finished.connect(self.finished_thread)

#--------------------Start prediction-----------------------------------------
    def predict(self, thread_count, log):
        self.pushpredict.setEnabled(False)
        #if self.runlocal.isChecked():
        run_local(self.config, destination='prediction')

        msg5 = QtWidgets.QMessageBox()
        msg5.setIcon(QtWidgets.QMessageBox.Information)
        msg5.setText("Prediction completed.")
        msg5.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg5.exec_()
        
        log.finished.emit("thread %d" % thread_count)

        self.pushpredict.setEnabled(True)

#--------------------End prediction-----------------------------------------

#--------------------Start prediction with trained model-----------------------------------------
    def predict_with_trained_model_thread(self):
        self.t4 = Thread(target=self.predict_with_trained_model, args=(self.counter_threads, self.l))
        self.t4.start()

        self.counter_threads += 1
        self.l.finished.connect(self.finished_thread)
        
    def predict_with_trained_model(self,  thread_count, log):
        
        self.config = dict()
        self.config['inputDir'] = self.inputdirtext_2.text()
        self.config['outputDir'] = self.outputdirtext_2.text()
        self.config['modelDir'] = self.modeldir_2.text()
        self.config['multichannel'] = False

        self.config['extension'] = self.pred_inputs['Input File Format Extension'].text()
        self.config['twoDim'] = self.pred_inputs['2D'].isChecked()
        

        print('predict_with_trained_model ::: ', self.config)

        run_local(self.config, destination='prediction_from_trained_model')

        log.finished.emit("thread %d" % thread_count)
#--------------------End prediction with trained model-----------------------------------------

#--------------------Start Validation of trained model-----------------------------------------
    def validation_thread(self):
        self.pushvalidate_5.setEnabled(False)
    
        self.t2=Thread(target=self.validate_trained_model, args=(self.counter_threads, self.l))
        self.t2.start()

        self.counter_threads += 1
        self.l.finished.connect(self.finished_thread)
        
    def validate_trained_model(self, thread_count, log):
        self.pushvalidate_5.setEnabled(False)
        self.config = dict()
        self.config['inputDir'] = self.inputdirtext_5.text()
        self.config['outputDir'] = self.outputdirtext_5.text()
        self.config['modelDir'] = self.modeldir_5.text()
        self.config['multichannel'] = False
        self.config['extension'] = self.val_inputs['Input File Format Extension'].text()
        self.config['twoDim'] = self.val_inputs['2D'].isChecked()

        print('Validating_trained_model ::: ', self.config)
        print('Running Locally')
        
        self.scores_dict = run_local(self.config, destination='validation')

        for i,j in zip(self.scores_dict['image_name'], self.scores_dict['iou']):
            self.validation_textbox.appendPlainText(str(i) + ' : ' + str(np.round(j, 3)))

        avg_iou = np.round(np.mean(self.scores_dict['iou']), 2)
        avg_prec = np.round(np.mean(self.scores_dict['precion']), 2)
        avg_rec = np.round(np.mean(self.scores_dict['recall']), 2)
        avg_acc = np.round(np.mean(self.scores_dict['accuracy']), 2)
        avg_dice = np.round(np.mean(self.scores_dict['dice']), 2)
        avg_auc = np.round(np.mean(self.scores_dict['auc']), 2)
        
        self.validation_textbox.appendPlainText('\n' + str(avg_iou) + ' : Overall IOU' +
                                                '\n' + str(avg_prec) + ' : Overall Precision' +
                                                '\n' + str(avg_rec) + ' : Overall Recall' +
                                                '\n' + str(avg_acc) + ' : Overall Accuracy' +
                                                '\n' + str(avg_dice) + ' : Overall Dice' +
                                                '\n' + str(avg_auc) + ' : Overall AUC')
            
        log.finished.emit("thread %d" % thread_count)
        self.pushval_csv.setEnabled(True)

    def save_validation_result(self):
        outputFilepath = os.path.join(self.config['outputDir'], 'Validation_result.csv')
        with open(outputFilepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.scores_dict.keys())

            # Write the data rows
            for row in zip(*self.scores_dict.values()):
                writer.writerow(row)
#--------------------End Validation of trained model-----------------------------------------

#--------------------Start Extraction-----------------------------------------
    def feature_extraction_thread(self):
        self.pushextract.setEnabled(False)
    
        self.t5 =Thread(target=self.feature_extraction, args=(self.counter_threads, self.l))
        self.t5.start()

        self.counter_threads += 1
        self.l.finished.connect(self.finished_thread)
        
    def feature_extraction(self, thread_count, log):

        self.config = dict()
        self.config['inputDir'] = self.inputdirtext_4.text()
        self.config['outputDir'] = self.outputdirtext_3.text()
        #self.config['runLocal'] = self.runlocal_extract.isChecked()
        self.config['nr_of_threads'] = self.extr_inputs['Number of Threads'].text()
        self.config['twoDim'] = self.extr_inputs['2D'].isChecked()
        print('Feature Extration parameters ::: ', self.config)

        f
        run_local(self.config, destination='feature_extraction')

        log.finished.emit("thread %d" % thread_count)
            
#--------------------End Extraction-----------------------------------------
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

# User name path
def user_path():
    homedir = os.path.expanduser("~")
    return homedir



app=QApplication(sys.argv)
mainwindow=MainWindow()

mainwindow.setMinimumWidth(1200)
mainwindow.setMinimumHeight(800)
mainwindow.setWindowTitle('StardistGUI')

print('Running StardistGUI version 0.1 alfa (build 2022-10-23) for', platform.system())
app.setWindowIcon(QIcon('stardist.ico'))
app.setApplicationName("Stardist GUI")
app.setWindowIcon(QIcon(os.path.join(user_path(), 'Documents/AI/StarDist_GUI/Imaging_tools-main_2022-09-11/static/stardist.icns'))) 

mainwindow.center()
mainwindow.show()
#widget.show()
sys.exit(app.exec_())
