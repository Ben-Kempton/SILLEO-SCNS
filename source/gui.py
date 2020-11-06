######################################################################
#                                                                    
# Part of SILLEO-SCNS, Provides the control GUI.
# Copyright (C) 2020  Benjamin S. Kempton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
######################################################################


# imports

# normal python libs
import sys
import math

import multiprocessing as mp
import threading as td

# engineering

# qt5 stuff
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from PyQt5.QtWidgets import QLineEdit, QGridLayout, QStyle, QCheckBox
from PyQt5.QtWidgets import QComboBox, QFileDialog

from simulation import Simulation


##############################################################################
# globals

# the mean radius of the earth in meters according to wikipedia
EARTH_RADIUS = 6371000

# if true, will enable calculating network link-state
MAKE_LINKS = True

# if you add a new network design (linking method)
# this list will need to be updated for it to appear
# in the GUI
NETWORK_DESIGNS = ["SPARSE", "+GRID", "IDEAL"]

# button styles:

# red, black text
STYLE_1 = 'QPushButton {background-color: rgba(255, 50, 50, 10); color: rgba(0,0,0,255);}'

# green, black text
STYLE_2 = 'QPushButton {background-color: rgba(50, 255, 50, 10); color: rgba(0,0,0,255);}'

# gray, black text
STYLE_3 = 'QPushButton {background-color: rgba(0, 0, 0, 50); color: rgba(0,0,0,5);}'

# gray, black text
STYLE_4 = 'QLabel {background-color: rgba(255, 255, 0, 150); color: rgba(0,0,0,255);}'

##############################################################################


class ApplicationWindow(QWidget):
	def __init__(self):
		super().__init__()
		self._main = QWidget()

		self.ctx = mp.get_context('spawn')

		self.topGrid = QGridLayout(self)

		# create input
		self.input = QGridLayout(self)
		self.makeInput()
		self.topGrid.addLayout(self.input, 0, 0)

		# create controls box
		self.controls = QGridLayout(self)
		self.makeControls()
		self.topGrid.addLayout(self.controls, 1, 0)

		self.makeConstellation()

	def closeEvent(self, event):
		print("called close func")
		try:
			self.vtkProcess.terminate()
		except AttributeError:
			print("Exiting...")
		except EOFError:
			print("Exiting...")
		event.accept()

	def makeInput(self):
		self.input.setSpacing(5)  # space between fields

		self.planesLabel = QLabel("Number of Planes:")
		self.planesEdit = QLineEdit("3")
		self.planesEdit.setToolTip('int')
		self.input.addWidget(self.planesLabel, 1, 0)
		self.input.addWidget(self.planesEdit,  1, 1)

		self.nodesLabel = QLabel("Number of Nodes/Plane:")
		self.nodesEdit = QLineEdit("12")
		self.nodesEdit.setToolTip("int")
		self.input.addWidget(self.nodesLabel, 2, 0)
		self.input.addWidget(self.nodesEdit,  2, 1)

		self.incLabel = QLabel("Plane inclination (deg):")
		self.incEdit = QLineEdit("65.0")
		self.incEdit.setToolTip('float')
		self.input.addWidget(self.incLabel, 3, 0)
		self.input.addWidget(self.incEdit,  3, 1)

		self.smaLabel = QLabel("Orbit Altitude (Km):")
		self.smaEdit = QLineEdit(str(int(500)))
		self.smaEdit.setToolTip('int')
		self.input.addWidget(self.smaLabel, 4, 0)
		self.input.addWidget(self.smaEdit,  4, 1)

	def makeControls(self):
		self.controls.setSpacing(5)  # space between buttons

		# make spawn simulation button
		self.generateButton = QPushButton("generate")
		self.generateButton.setToolTip("spawn a new simulation process")
		self.generateButton.setStyleSheet(STYLE_2)
		self.controls.addWidget(self.generateButton, 0, 0)
		self.generateButton.clicked.connect(self.makeVTKModel)

		# make kill simulation process button
		self.killButton = QPushButton("kill")
		self.killButton.setToolTip("kill the last created simulation process")
		self.killButton.setStyleSheet(STYLE_1)
		self.controls.addWidget(self.killButton, 0, 1)
		self.killButton.clicked.connect(self.killVTKModel)

		# make play/pause button
		self.pause = True
		self.togglePlayButton = QPushButton()
		self.togglePlayButton.setToolTip("pause/run the simulation process")
		self.togglePlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
		self.togglePlayButton.setStyleSheet(STYLE_3)
		self.togglePlayButton.setEnabled(True)
		self.controls.addWidget(self.togglePlayButton, 0, 2)
		self.togglePlayButton.clicked.connect(self.pauseStart)

		# make import model from gml button
		self.importGMLButton = QPushButton("Import GML")
		self.importGMLButton.setToolTip("spawn a new simulation process from a GML file")
		self.importGMLButton.setStyleSheet(STYLE_2)
		self.controls.addWidget(self.importGMLButton, 0, 3)
		self.importGMLButton.clicked.connect(self.genModelFromGML)

		# make timestep setter
		self.timestepLabel = QLabel('Timestep(seconds):')
		self.timestepEdit = QLineEdit(str(int(10)))  # 30 second default timestep
		self.timestepEdit.setToolTip("int : simulation process timestep")
		self.timestepButton = QPushButton("set")
		self.timestepButton.setToolTip("must press to update simulation")
		self.timestepButton.clicked.connect(self.setTimestep)
		self.controls.addWidget(self.timestepLabel, 1, 0)
		self.controls.addWidget(self.timestepEdit, 1, 1)
		self.controls.addWidget(self.timestepButton, 1, 2)

		# make label to show current time
		self.timeLabel = QLabel("Time: 0:00:00")
		self.timeLabel.setToolTip("current time of simulation, updated live")
		self.timeLabel.setStyleSheet(STYLE_4)
		self.controls.addWidget(self.timeLabel, 2, 0)

		# make checkbox for toggle image capture
		self.saveImagesLabel = QLabel("Save Images:")
		self.saveImagesCheckbox = QCheckBox()
		self.saveImagesCheckbox.setToolTip('check this box to begin saving images\npath: ./pics')
		self.saveImagesCheckbox.stateChanged.connect(self.toggleSaveImages)
		self.controls.addWidget(self.saveImagesLabel, 3, 0)
		self.controls.addWidget(self.saveImagesCheckbox, 3, 1)

		# make checkbox for toggle image capture
		self.saveGMLSLabel = QLabel("Save GMLs:")
		self.saveGMLSCheckbox = QCheckBox()
		self.saveGMLSCheckbox.setToolTip('check this box to begin saving GML files\npath: ./gmls')
		self.saveGMLSCheckbox.stateChanged.connect(self.toggleSaveGMLS)
		self.controls.addWidget(self.saveGMLSLabel, 4, 0)
		self.controls.addWidget(self.saveGMLSCheckbox, 4, 1)

		# add a drop down to select network design method
		self.linksLabel = QLabel("Network Design:")
		self.linksSelect = QComboBox()
		self.linksSelect.addItems(NETWORK_DESIGNS)
		self.linksSelect.activated[str].connect(self.setLinkingMethod)
		self.controls.addWidget(self.linksLabel, 5, 0)
		self.controls.addWidget(self.linksSelect, 5, 1)

		# controls for running for a set time
		self.runforLabel = QLabel("Run For (seconds):")
		self.runforEdit = QLineEdit(str(int(60)))
		self.runforEdit.setToolTip('int')
		self.runforButton = QPushButton("Run!")
		self.runforButton.setToolTip('click to run the simulation for a set time')
		self.runforButton.clicked.connect(self.setRunfor)
		self.controls.addWidget(self.runforLabel, 6, 0)
		self.controls.addWidget(self.runforEdit, 6, 1)
		self.controls.addWidget(self.runforButton, 6, 2)

		# section for setting up a connection
		self.node1Label = QLabel("Node 1:")
		self.node1Select = QComboBox()
		self.node1Select.activated[str].connect(self.setPathNode1)
		self.node2Label = QLabel("Node 2:")
		self.node2Select = QComboBox()
		self.node2Select.activated[str].connect(self.setPathNode2)
		self.calcPathButton = QPushButton("Enable Path Calculation")
		self.calcPathButton.setCheckable(True)
		self.calcPathButton.clicked.connect(self.togglePathCalculation)
		self.controls.addWidget(self.node1Label, 7, 0)
		self.controls.addWidget(self.node1Select, 7, 1)
		self.controls.addWidget(self.node2Label, 7, 2)
		self.controls.addWidget(self.node2Select, 7, 3)
		self.controls.addWidget(self.calcPathButton, 8, 0)

		# section for showing performance data
		self.frameTimeLabel = QLabel("---")
		self.modelTimeLabel = QLabel("---")
		self.gmlTimeLabel = QLabel("---")
		self.renderTimeLabel = QLabel("---")
		self.imgTimeLabel = QLabel('---')
		self.maxNodeDegreeLabel = QLabel('---')
		self.totalNumLinksLabel = QLabel('---')
		self.controls.addWidget(self.frameTimeLabel, 10, 0)
		self.controls.addWidget(self.modelTimeLabel, 11, 0)
		self.controls.addWidget(self.renderTimeLabel, 12, 0)
		self.controls.addWidget(self.gmlTimeLabel, 13, 0)
		self.controls.addWidget(self.imgTimeLabel, 14, 0)
		self.controls.addWidget(self.maxNodeDegreeLabel, 15, 0)
		self.controls.addWidget(self.totalNumLinksLabel, 16, 0)

		# a debugging test button
		self.testButton = QPushButton("test")
		self.testButton.setToolTip('a debugging test button')
		self.controls.addWidget(self.testButton, 20, 0)
		self.testButton.clicked.connect(self.testFunc)

	def genModelFromGML(self):
		filename = QFileDialog.getOpenFileName(self)
		print("importing the file: ", filename[0])

		# make sure any existing models are killed
		try:
			self.vtkProcess.terminate()
		except AttributeError:
			print("...")
		except EOFError:
			print("...")

		# a pipe to allow communication between processes
		self.myPipeConn, otherPipeConn = self.ctx.Pipe()
		kwargsToSend = {
			'pipeConn': otherPipeConn,
			'animate': True,
			'gmlImportFileName': filename[0]
		}
		print(kwargsToSend)

		self.vtkProcess = self.ctx.Process(target=Simulation, kwargs=kwargsToSend)
		self.vtkProcess.start()

		# start a thread to listen for stuff from the model
		self.comsThread = td.Thread(target=self.comsThreadHandler)
		self.comsThread.start()

	def makeConstellation(self):

		# gen a constellation object
		p = int(self.planesEdit.text())
		n = int(self.nodesEdit.text())
		i = float(self.incEdit.text())
		a = float(self.smaEdit.text())*1000 + EARTH_RADIUS
		ts = int(self.timestepEdit.text())
		self.con = {
			'planes': p,
			'nodesPerPlane': n,
			'inclination': i,
			'semiMajorAxis': a,
			'timeStep': ts
		}

	def killVTKModel(self):
		try:
			self.vtkProcess.terminate()
		except EOFError:
			print("ERROR: tried to kill vtk process, but something went wrong")

	def makeVTKModel(self):
		# make sure any existing models are killed
		try:
			self.vtkProcess.terminate()
		except AttributeError:
			print("...")
		except EOFError:
			print("...")

		# be sure to grab latest input params
		self.makeConstellation()

		# a pipe to allow communication between processes
		self.myPipeConn, otherPipeConn = self.ctx.Pipe()
		kwargsToSend = {'pipeConn': otherPipeConn, 'makeLinks': MAKE_LINKS}
		kwargsToSend.update(self.con)  # add the self.con dict to kwargs dict
		self.vtkProcess = self.ctx.Process(target=Simulation, kwargs=kwargsToSend)
		self.vtkProcess.start()

		# start a thread to listen for stuff from the model
		self.comsThread = td.Thread(target=self.comsThreadHandler)
		self.comsThread.start()

	def comsThreadHandler(self):
		# now I just listen for data coming
		# back from the simulation
		while True:
			try:
				recv = self.myPipeConn.recv()
				if type(recv) == str:
					print(recv)

				elif type(recv) == list:
					command = recv[0]
					if command == "simTime":
						time = recv[1]
						hours = int(time / 3600)
						time = time % 3600
						mins = int(time / 60)
						secs = int(time % 60)
						time = str(hours)+":"+str(mins)+":"+str(secs)
						self.timeLabel.setText("Time: "+time)
					elif command == 'timeForFrame':
						self.frameTime = recv[1]
						temp = 1.0 / recv[1]
						self.frameTimeLabel.setText(str("%3.3f Hz" % temp))
					elif command == 'timeToUpdateModel':
						temp = (recv[1]/(self.frameTime))*100
						self.modelTimeLabel.setText('sim: %3.2f%%' % temp)
					elif command == 'timeToExportGML':
						temp = recv[1]/(self.frameTime)*100
						self.gmlTimeLabel.setText("gml: %3.2f%%" % temp)
					elif command == 'timeToUpdateRender':
						temp = recv[1]/(self.frameTime)*100
						self.renderTimeLabel.setText("ren: %3.2f%%" % temp)
					elif command == 'timeToExportImg':
						temp = recv[1]/(self.frameTime)*100
						self.imgTimeLabel.setText("img: %3.2f%%" % temp)
					elif command == 'maxNodeDegree':
						temp = str(recv[1])
						self.maxNodeDegreeLabel.setText('Max node degree: ' + temp)
					elif command == 'totalNumberOfLinks':
						temp = str(recv[1])
						self.totalNumLinksLabel.setText('Total Links (edges): ' + temp)
					elif command == 'placeNames':
						self.placenames = recv[1]
						self.node1Select.clear()
						self.node2Select.clear()
						self.node1Select.addItems(recv[1])
						self.node2Select.addItems(recv[1])

			except EOFError:
				break

	def pauseStart(self):
		self.pause = not self.pause
		if not self.pause:
			self.togglePlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
		else:
			self.togglePlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
		try:
			self.myPipeConn.send("togglePause")
		except EOFError:
			print('pause button did not work')
		except AttributeError:
			print('no model to pause...')

	def setTimestep(self):
		timestep = int(self.timestepEdit.text())
		temp = ["setTimestep", timestep]
		self.myPipeConn.send(temp)

	def setLinkingMethod(self, text):
		self.myPipeConn.send(["setLinkingMethod", text])

	def setRunfor(self):
		"""
		runs the simulation for a set time then stops
		"""

		runtime = int(self.runforEdit.text())
		self.pause = False
		self.togglePlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
		try:
			self.myPipeConn.send(['setRunfor', runtime])
		except EOFError:
			print('no model')

	def togglePathCalculation(self):
		if self.calcPathButton.isChecked():
			try:
				self.myPipeConn.send("enablePathCalc")
			except EOFError:
				print('no model')
		else:
			try:
				self.myPipeConn.send("disablePathCalc")
			except EOFError:
				print('no model')

	def setPathNode1(self, text):
		self.myPipeConn.send(["setPathNode1", text])

	def setPathNode2(self, text):
		self.myPipeConn.send(["setPathNode2", text])

	def toggleSaveGMLS(self):
		if self.saveGMLSCheckbox.isChecked():
			self.enableGMLCapture()
		else:
			self.disableGMLCapture()

	def enableGMLCapture(self):
		try:
			self.myPipeConn.send("enableGMLCapture")
		except EOFError:
			print('enable capture did not work')

	def disableGMLCapture(self):
		try:
			self.myPipeConn.send("disableGMLCapture")
		except EOFError:
			print("disable capture did not work")

	def toggleSaveImages(self):
		if self.saveImagesCheckbox.isChecked():
			self.enableImageCapture()
		else:
			self.disableImageCapture()

	def enableImageCapture(self):
		try:
			self.myPipeConn.send("enableImageCapture")
		except EOFError:
			print('enable capture did not work')

	def disableImageCapture(self):
		try:
			self.myPipeConn.send("disableImageCapture")
		except EOFError:
			print("disable capture did not work")

	def testFunc(self):
		try:
			gndNodeInfo = ['newGndNode', [1.0, 0, 0]]
			self.myPipeConn.send(gndNodeInfo)
		except EOFError:
			print('test button did not work')

	def makeGroundPoint(self):
		# try:
		lat = float(self.latEdit.text())
		lon = float(self.lonEdit.text())
		xyz = self.latLonToXYZ(lat, lon)
		self.myPipeConn.send(['newGndNode', xyz])

	def latLonToXYZ(self, lat, lon):
		x = EARTH_RADIUS * math.cos(lat) * math.cos(lon)
		y = EARTH_RADIUS * math.cos(lat) * math.sin(lon)
		z = EARTH_RADIUS * math.sin(lat)
		return [x, y, z]


if __name__ == "__main__":
	qt_app = QApplication(sys.argv)
	app = ApplicationWindow()
	app.show()
	qt_app.exec_()
