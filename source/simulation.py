######################################################################
#                                                                    
# Part of SILLEO-SCNS, high level simulation control and visualization
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

# used to make program faster & responsive
import threading as td

# memory aligned arrays their manipulation for Python
import numpy as np

# custom classes
from constellation import Constellation

# OpenGL API for Python
import vtk

# Primarily using the write_gml() function...
import networkx as nx

# use to measure program performance (sim framerate)
import time

# try to import numba funcs
try:
	import numba_funcs as nf
	USING_NUMBA = True
except ModuleNotFoundError:
	USING_NUMBA = False
	print("you probably do not have numba installed...")
	print("reverting to non-numba mode")


###############################################################################
#                               GLOBAL VARS                                   #
###############################################################################

EARTH_RADIUS = 6371000  # radius of Earth in meters

PNG_OUTPUT_PATH = "pics/p"  # where to save images of the animation
GML_OUTPUT_PATH = 'gmls/g'  # where to save gml files

MIN_SAT_ELEVATION = 30  # degrees

LANDMASS_OUTLINE_COLOR = (0.0, 0.0, 0.0)  # black, best contrast
EARTH_LAND_OPACITY = 1.0

EARTH_BASE_COLOR = (0.6, 0.6, 0.8)  # light blue, like water!
EARTH_OPACITY = 1.0

BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # white

SAT_COLOR = (1.0, 0.0, 0.0)  # red, color of satellites
SAT_OPACITY = 1.0

GND_COLOR = (0.0, 1.0, 0.0)  # green, color of groundstations
GND_OPACITY = 1.0

ISL_LINK_COLOR = (0.9, 0.5, 0.1)  # yellow-brown, satellite-satellite links
ISL_LINK_OPACITY = 1.0
ISL_LINE_WIDTH = 3  # how wide to draw line in pixels

SGL_LINK_COLOR = (0.5, 0.9, 0.5)  # greenish? satellite-groundstation links
SGL_LINK_OPACITY = 0.75
SGL_LINE_WIDTH = 2  # how wide to draw line in pixels

PATH_LINK_COLOR = (0.8, 0.2, 0.8)  # purpleish? path links
PATH_LINK_OPACITY = 0.7
PATH_LINE_WIDTH = 13  # how wide to draw line in pixels

EARTH_SPHERE_POINTS = 5000  # higher = smoother earth model, slower to generate

SAT_POINT_SIZE = 9  # how big satellites are in (probably) screen pixels
GND_POINT_SIZE = 8  # how big ground points are in (probably) screen pixels

SECONDS_PER_DAY = 86400  # number of seconds per earth rotation (day)


def getFileNumber(var):
	"""
	Makes a nice int with leading zeros for file naming

	Attributes
	----------
	var : int
		an int that will be used to label export files

	Returns
	-------
	string : string
		a 7 char int, using leading zeros
	"""

	mask = ['0', '0', '0', '0', '0', '0', '0']
	var = list(str(var))
	t = len(mask) - len(var)
	for i in range(len(var)):
		mask[i+t] = var[i]
	return ''.join(mask)


###############################################################################
#                             SIMULATION CONTROL                              #
###############################################################################


class Simulation():

	def __init__(
			self,
			pipeConn=None,
			planes=1,
			nodesPerPlane=1,
			inclination=70,
			semiMajorAxis=6472000,
			timeStep=10,
			makeLinks=True,
			animate=True,
			captureImages=False,
			captureInterpolation=1,
			captureGML=False,
			groundPtsFile='city_data.txt',
			gmlImportFileName=None):

		if gmlImportFileName is not None:
			# try to import the given file as a networkX graph
			try:
				G = nx.read_gml(gmlImportFileName)
			except FileNotFoundError as error:
				print("ERROR! tried to import gml file, file not found\n", error)
				return None

			# assuming gml import worked, we extract the simulation
			# structure data like time, num-planes, inclination... etc
			# constillation structure information
			self.num_planes = int(G.graph['numPlanes'])
			self.num_nodes_per_plane = int(G.graph['numNodesPerPlane'])
			self.plane_inclination = float(G.graph['planeInclination'])
			self.semi_major_axis = int(float(G.graph['semiMajorAxisMeters']))
			self.min_communications_altitude = int(G.graph['minCommunicationsAltitudeMeters'])
			self.min_sat_elevation = int(G.graph['minSatElevationDegrees'])

			# path calculation
			self.path_node_1 = None
			self.path_node_2 = None
			self.path_length = 0.0
			self.path_links = None
			self.max_node_degree = -1

			# control flags
			self.animate = True
			self.capture_gml = False
			self.capture_images = False
			self.capt_interpolation = 1
			self.make_links = True
			self.linking_method = '+GRID'  # used because it does not regenerate links
			self.enable_path_calculation = False

			# timing control
			self.time_step = 1
			self.current_simulation_time = int(float(G.graph['simulationTime']))
			self.pause = True
			self.num_steps_to_run = -1

			# performance data
			self.time_1 = 1
			self.time_for_frame = 1
			self.time_to_update_model = 1
			self.time_to_export_gml = 1
			self.time_to_update_render = 1
			self.time_to_export_img = 1

			# get all of the edge data
			edge_data = list(G.edges())
			# returns in data in string format, so I convert data to ints
			for i in range(len(edge_data)):
				edge_data[i] = [int(edge_data[i][0]), int(edge_data[i][1])]

			# init the 'pipe' object used for inter-process communication
			# this comes from the multiprocessing library
			self.pipeConn = pipeConn
			self.controlThread = td.Thread(target=self.controlThreadHandler)
			self.controlThread.start()

			# init the Constellation model
			self.model = Constellation(
				planes=self.num_planes,
				nodes_per_plane=self.num_nodes_per_plane,
				inclination=self.plane_inclination,
				semi_major_axis=self.semi_major_axis,
				minCommunicationsAltitude=self.min_communications_altitude,
				minSatElevation=self.min_sat_elevation,
				linkingMethod=self.linking_method)

			# add ground points to the constillation model
			# from the given file path
			# TODO:add error protection...
			data = []
			self.city_names = []
			with open(groundPtsFile, 'r') as f:
				for line in f:
					my_line = []
					for word in line.split():
						my_line.append(word)
					data.append(my_line)
			for i in range(1, len(data)):
				self.city_names.append(data[i][0])
				self.model.addGroundPoint(float(data[i][1]), float(data[i][2]))

			# send the names to the GUI so the user
			# can see them in drop down menu
			self.pipeConn.send(["placeNames", self.city_names])

			# init the network design
			if self.make_links:
				print("test")
				self.max_isl_distance = self.model.calculateMaxISLDistance(
					self.min_communications_altitude)
				print(self.max_isl_distance)

				self.max_stg_distance = self.model.calculateMaxSpaceToGndDistance(
					self.min_sat_elevation)

				self.model.import_links_from_gml_data(edge_data)

				self.model.setConstillationTime(self.current_simulation_time)

				self.model.calculatePlusGridLinks(self.max_stg_distance, max_isl_range=self.max_isl_distance)

			print('test2')
			# so, after much effort it appears that I cannot control an
			# interactive vtk window externally. Therefore when running
			# with an animation, the animation class will have to drive
			# the simulation using an internal timer...
			if self.animate:
				self.setupAnimation(
					self.model.total_sats,
					self.model.getArrayOfSatPositions(),
					-self.model.ground_node_counter,
					self.model.getArrayOfGndPositions(),
					self.time_step,
					self.current_simulation_time,
					self.capture_images
				)

				print('test3')

			else:
				# TODO:then we run drive the simulation from here
				return

		else:  # if gmlImportFileName is None

			# constillation structure information
			self.num_planes = planes
			self.num_nodes_per_plane = nodesPerPlane
			self.plane_inclination = inclination
			self.semi_major_axis = semiMajorAxis
			self.min_communications_altitude = 100000
			self.min_sat_elevation = MIN_SAT_ELEVATION

			# path calculation
			self.path_node_1 = None
			self.path_node_2 = None
			self.path_length = 0.0
			self.path_links = None
			self.max_node_degree = -1

			# control flags
			self.animate = animate
			self.capture_gml = captureGML
			self.capture_images = captureImages
			self.capt_interpolation = 1
			self.make_links = makeLinks
			self.linking_method = 'SPARSE'  # options: 'IDEAL', '+GRID', 'SPARSE'
			self.enable_path_calculation = False

			# timing control
			self.time_step = timeStep
			self.current_simulation_time = 0.0
			self.pause = True
			self.num_steps_to_run = -1

			# performance data
			self.time_1 = 1
			self.time_for_frame = 1
			self.time_to_update_model = 1
			self.time_to_export_gml = 1
			self.time_to_update_render = 1
			self.time_to_export_img = 1

			# init the 'pipe' object used for inter-process communication
			# this comes from the multiprocessing library
			self.pipeConn = pipeConn
			self.controlThread = td.Thread(target=self.controlThreadHandler)
			self.controlThread.start()

			# init the Constellation model
			self.model = Constellation(
				planes=self.num_planes,
				nodes_per_plane=self.num_nodes_per_plane,
				inclination=self.plane_inclination,
				semi_major_axis=self.semi_major_axis,
				minCommunicationsAltitude=self.min_communications_altitude,
				minSatElevation=self.min_sat_elevation,
				linkingMethod=self.linking_method)

			# add ground points to the constillation model
			# from the given file path
			# TODO:add error protection...
			data = []
			self.city_names = []
			with open(groundPtsFile, 'r') as f:
				for line in f:
					my_line = []
					for word in line.split():
						my_line.append(word)
					data.append(my_line)
			for i in range(1, len(data)):
				self.city_names.append(data[i][0])
				self.model.addGroundPoint(float(data[i][1]), float(data[i][2]))

			# send the names to the GUI so the user
			# can see them in drop down menu
			self.pipeConn.send(["placeNames", self.city_names])

			# init the network design
			if self.make_links:
				self.initializeNetworkDesign()

			# so, after much effort it appears that I cannot control an
			# interactive vtk window externally. Therefore when running
			# with an animation, the animation class will have to drive
			# the simulation using an internal timer...
			if self.animate:
				self.setupAnimation(
					self.model.total_sats,
					self.model.getArrayOfSatPositions(),
					-self.model.ground_node_counter,
					self.model.getArrayOfGndPositions(),
					self.time_step,
					self.current_simulation_time,
					self.capture_images
				)

			else:
				# TODO:then we run drive the simulation from here
				return

	def initializeNetworkDesign(self):
		print("initalizing network design... ")
		self.max_isl_distance = self.model.calculateMaxISLDistance(
			self.min_communications_altitude)

		self.max_stg_distance = self.model.calculateMaxSpaceToGndDistance(
			self.min_sat_elevation)

		print('maxIsl: ', self.max_isl_distance)
		print('maxGtS: ', self.max_stg_distance)

		if self.linking_method == 'IDEAL':
			self.model.calculateIdealLinks(
				self.max_isl_distance,
				self.max_stg_distance)

		if self.linking_method == '+GRID':
			self.model.calculatePlusGridLinks(
				self.max_stg_distance,
				initialize=True,
				crosslink_interpolation=1)

		if self.linking_method == 'SPARSE':
			self.model.calculatePlusGridLinks(
				self.max_stg_distance,
				initialize=True,
				crosslink_interpolation=self.model.total_sats + 1)

		print("done initalizing")

	def controlThreadHandler(self):
		"""
		Start a thread to deal with inter-process communications

		"""

		while True:
			received_data = self.pipeConn.recv()
			if type(received_data) == str:
				if received_data == "doTestFunc":
					self.testFunc()

				if received_data == "enableImageCapture":
					print("enabled image capture")
					self.capture_images = True

				if received_data == "disableImageCapture":
					print("disabled image capture")
					self.capture_images = False

				if received_data == "enableGMLCapture":
					print("enabled GML capture")
					self.capture_gml = True

				if received_data == "disableGMLCapture":
					print("disabled GML capture")
					self.capture_gml = False

				if received_data == "toggleLinks":
					self.make_links = not self.make_links

				if received_data == "togglePause":
					self.pause = not self.pause
					print('toggle pause recived by sim, pause is: ', self.pause)

				if received_data == "enablePathCalc":
					self.enable_path_calculation = True
					print("enabled path calculation")

				if received_data == "disablePathCalc":
					self.enable_path_calculation = False
					print("disabled path calculation")

			elif type(received_data) == list:
				command = received_data[0]
				if command == "setTimestep":
					print("setting timestep to: ", received_data[1])
					self.time_step = received_data[1]

				if command == "setLinkingMethod":
					self.linking_method = received_data[1]
					print("set linking method to: ", received_data[1])
					# if reset linking method, must reinit network
					self.initializeNetworkDesign()

				if command == "setRunfor":
					self.num_steps_to_run = int(received_data[1] / self.time_step)-1
					print("running for: ", self.num_steps_to_run, " timesteps...")
					self.pause = False

				if command == 'setPathNode1':
					self.path_node_1 = received_data[1]
					print("set path node 1: ", self.path_node_1)

				if command == 'setPathNode2':
					self.path_node_2 = received_data[1]
					print("set path node 2: ", self.path_node_2)

			else:
				print(received_data)

	def statusReport(self):
		"""
		sends some status data like current time back to host process

		"""

		self.pipeConn.send(["simTime", self.current_simulation_time])
		self.pipeConn.send(['timeForFrame', self.time_for_frame])
		self.pipeConn.send(['timeToUpdateModel', self.time_to_update_model])
		self.pipeConn.send(['timeToExportGML', self.time_to_export_gml])
		self.pipeConn.send(['timeToUpdateRender', self.time_to_update_render])
		self.pipeConn.send(['timeToExportImg', self.time_to_export_img])
		self.pipeConn.send(['maxNodeDegree', self.max_node_degree])
		self.pipeConn.send(['totalNumberOfLinks', self.model.total_links])

	def testFunc(self):
		"""
		Test function for debugging.
		"""

		print("this is the vtk test func being run")
		# print(self.num_planes)
		# self.updateVtkSatPos()
		# for name, node in self.conn.nodes.items():
		#    print(name, node)

		self.pipeConn.send("back to you")
		return

	def updateModel(self, new_time):
		"""
		Update the model with a new time, recalculate links, & export GML files

		Function behaves differently depending on wether animate is true or not.
		If true, this func will be called from the updateAnimation() func
		If False, this will be called in a loop until some desired runtime is reached

		"""

		# grab initial time
		self.time_1 = time.time()

		self.frameCount += 1

		if self.num_steps_to_run > 0:
			self.num_steps_to_run -= 1
		elif self.num_steps_to_run == 0:
			self.pause = True
			self.num_steps_to_run = -1

		# update links / network design / sat positions
		self.model.setConstillationTime(new_time)
		if self.make_links:
			if self.linking_method == 'IDEAL':
				self.model.calculateIdealLinks(
					self.max_isl_distance,
					self.max_stg_distance)
			if self.linking_method == '+GRID':
				self.model.calculatePlusGridLinks(self.max_stg_distance, max_isl_range=self.max_isl_distance)
			if self.linking_method == 'SPARSE':
				self.model.calculatePlusGridLinks(self.max_stg_distance)

		self.time_to_update_model = time.time() - self.time_1

		# generate GML graph if applicable
		if self.capture_gml and \
		   self.frameCount % self.capt_interpolation == 0 or \
		   self.enable_path_calculation:
			# set up graph, recording important info about the current model
			self.model.generateNetworkGraph(self.city_names)

			# calculate max node degree (number of edges connected to a node)
			degree_list = list(self.model.G.degree())
			for i in range(len(degree_list)):
				degree_list[i] = int(degree_list[i][1])
			self.max_node_degree = max(degree_list)

		# if enabled, we run dijkstra's between two points
		if self.enable_path_calculation:
			node_1 = self.path_node_1
			node_2 = self.path_node_2
			if (node_1 is not None) and (node_2 is not None):
				id_1 = -(self.city_names.index(node_1) + 1)
				id_2 = -(self.city_names.index(node_2) + 1)

				# run shortest path, handle exception if path not exist
				try:
					path = nx.shortest_path(
						self.model.G,
						source=str(id_1),
						target=str(id_2),
						weight='distance')

					# convert list of nodes into edges
					self.path_links = []
					for i in range(len(path)-1):
						self.path_links.append([path[i], path[i+1]])

				except nx.exception.NetworkXNoPath:
					print("path does not exist...")
					self.path_links = None

		# TODO:figure out the max number of links per sat in ideal case

		if self.capture_gml and self.frameCount % self.capt_interpolation == 0:
			file_name = GML_OUTPUT_PATH+"_"+getFileNumber(self.frameCount)+".gml"
			self.model.exportGMLFile(file_name)

		self.time_to_export_gml = time.time() -\
		self.time_1 - self.time_to_update_model

###############################################################################
#                           ANIMATION FUNCTIONS                               #
###############################################################################

	"""
	Like me, you might wonder what the numerous vkt calls are for.
	Answer: you need to manually configure a render pipeline for
	each object (vtk actor) in the scene.
	A typical VTK render pipeline:

	point data array   <-- set/update position data
	    |
	poly data array
	    |
	poly data mapper
	    |
	object actor   <-- edit color/size/opacity, apply rotations/translations
	    |
	vtk renderer
	    |
	vkt render window
	vkt render interactor   <-- trigger events, animate
	    |
	Your computer screen
	exported png files

	"""

	def setupAnimation(
			self,
			total_satellites,
			satellite_positions,
			total_groundpoints,
			groundpoint_positions,
			timestep=60,
			current_simulation_time=0,
			capture_images=False):
		"""
		Makes vtk render window, and sets up pipelines.

		Parameters
		----------
		total_satellites : int
			The total number of satellties in the model
		satellite_positions : np.array[[('x', int32), ('y', int32), ('z', int32)]]
			Numpy array of all the satellite positions
		total_groundpoints : int
			Total number of groundpoints in the model
		groundpoint_positions : np.array[[('x', int32), ('y', int32), ('z', int32)]]
			Numpy array of all the groundpoint positions
		timestep : int
			Timestep for the simulation in seconds
		current_simulation_time : int
			current time of the simulation in seconds
		capture_images : bool
			If true, save images of the render window to file

		"""

		self.current_simulation_time = current_simulation_time
		self.time_step = timestep
		self.capture_images = capture_images

		self.frameCount = 0
		self.incFrameCount = 1

		self.makeEarthActors(EARTH_RADIUS)

		if total_satellites > 0:
			self.makeSatsActor(total_satellites, satellite_positions)

		if total_groundpoints > 0:
			self.makeGndPtsActor(total_groundpoints, groundpoint_positions)

		if self.make_links:
			self.makeLinkActors()

		self.makeRenderWindow()

	def updateAnimation(self, obj, event):
		"""
		This function takes in new position data and updates the render window

		Parameters
		----------
		obj : ?
			The object that generated the event, probably vtk render window
		event : event
			The event that triggered this function
		"""

		if not self.pause:
			# update simulation time
			new_simulation_time = self.current_simulation_time + self.time_step

			# call func to update the model
			self.updateModel(new_simulation_time)

			# rotate earth and land
			rotation_per_time_step = 360.0/(SECONDS_PER_DAY / self.time_step)
			self.earthActor.RotateZ(rotation_per_time_step)
			self.sphereActor.RotateZ(rotation_per_time_step)

		else:
			new_simulation_time = self.current_simulation_time

		# grab new position data
		new_sat_positions = self.model.getArrayOfSatPositions()
		new_groundpoint_positions = self.model.getArrayOfGndPositions()

		# update sat points
		for i in range(self.totalSats):
			x = new_sat_positions[i]['x']
			y = new_sat_positions[i]['y']
			z = new_sat_positions[i]['z']
			self.satVtkPts.SetPoint(self.satPointIDs[i], x, y, z)
		self.satPolyData.GetPoints().Modified()

		# update gnd pts
		for i in range(self.totalGroundpoints):
			x = new_groundpoint_positions[i]['x']
			y = new_groundpoint_positions[i]['y']
			z = new_groundpoint_positions[i]['z']
			self.gndVtkPts.SetPoint(self.gndPointIDs[i], x, y, z)
		self.gndPolyData.GetPoints().Modified()

		# print('just before func')
		if self.make_links:

			# grab the arrays of connections
			links = self.model.getArrayOfLinks()
			points = self.model.getArrayOfNodePositions()
			maxSatIdx = self.model.total_sats-1

			# build a vtkPoints object from array
			self.linkPoints = vtk.vtkPoints()
			self.linkPoints.SetNumberOfPoints(len(points))
			for i in range(len(points)):
				self.linkPoints.SetPoint(i, points[i]['x'], points[i]['y'], points[i]['z'])

			# make clean line arrays
			self.islLinkLines = vtk.vtkCellArray()
			self.sglLinkLines = vtk.vtkCellArray()
			self.pathLinkLines = vtk.vtkCellArray()

			# fill isl and gsl arrays
			for i in range(len(links)):
				e1 = links[i]['node_1']
				e2 = links[i]['node_2']
				# must translate link endpoints to point names
				# if endpoint name is positive, we use it directly
				# if negative, idx = maxSatIdx-endpointname
				# **ground endpoints are always node_1**
				if e1 < 0:
					self.sglLinkLines.InsertNextCell(2)
					self.sglLinkLines.InsertCellPoint(maxSatIdx-e1)
					self.sglLinkLines.InsertCellPoint(e2)
				else:
					self.islLinkLines.InsertNextCell(2)
					self.islLinkLines.InsertCellPoint(e1)
					self.islLinkLines.InsertCellPoint(e2)

			self.sglPolyData.SetPoints(self.linkPoints)
			self.sglPolyData.SetLines(self.sglLinkLines)
			self.islPolyData.SetPoints(self.linkPoints)
			self.islPolyData.SetLines(self.islLinkLines)

			if self.enable_path_calculation and (self.path_links is not None):
				for link in self.path_links:
					e1 = int(link[0])
					e2 = int(link[1])
					self.pathLinkLines.InsertNextCell(2)
					if e1 < 0:
						self.pathLinkLines.InsertCellPoint(maxSatIdx-e1)
					else:
						self.pathLinkLines.InsertCellPoint(e1)
					if e2 < 0:
						self.pathLinkLines.InsertCellPoint(maxSatIdx-e2)
					else:
						self.pathLinkLines.InsertCellPoint(e2)

			self.pathPolyData.SetPoints(self.linkPoints)
			self.pathPolyData.SetLines(self.pathLinkLines)

		obj.GetRenderWindow().Render()
		self.current_simulation_time = new_simulation_time
		if not self.pause:
			self.time_to_update_render = (time.time() - self.time_1) -\
			(self.time_to_update_model + self.time_to_export_gml)

		if self.capture_images \
		   and self.frameCount % self.capt_interpolation == 0\
		   and not self.pause:
			self.renderToPng()

		if not self.pause:
			self.time_to_export_img = (time.time() - self.time_1) -\
			(self.time_to_update_model + self.time_to_export_gml +
			 self.time_to_update_render)

			self.time_for_frame = time.time() - self.time_1
			self.statusReport()

	def renderToPng(self, path=PNG_OUTPUT_PATH):
		"""
		Take a .png of the render window, and save it.

		Parameters
		----------
		path : str
		The relative path of where to save the image

		"""

		# make sure the path exists
		if not path:
			return

		# connect the image writer to the render window
		w2i = vtk.vtkWindowToImageFilter()
		w2i.SetInputBufferTypeToRGBA()
		w2i.SetInput(self.renderWindow)
		w2i.Update()
		pngfile = vtk.vtkPNGWriter()
		pngfile.SetInputConnection(w2i.GetOutputPort())

		# name the file with 7 digit int, leading zeros
		pngfile.SetFileName(path + "_" + getFileNumber(self.frameCount) + ".png")
		pngfile.Write()

	def makeRenderWindow(self):
		"""
		Makes a render window object using vtk.

		This should not be called until all the actors are created.

		"""

		# create a renderer object
		self.renderer = vtk.vtkRenderer()
		self.renderWindow = vtk.vtkRenderWindow()
		self.renderWindow.AddRenderer(self.renderer)

		# create an interactor object, to interact with the window... duh
		self.interactor = vtk.vtkRenderWindowInteractor()
		self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
		self.interactor.SetRenderWindow(self.renderWindow)

		# add the actor objects
		self.renderer.AddActor(self.satsActor)
		self.renderer.AddActor(self.earthActor)
		self.renderer.AddActor(self.gndActor)
		self.renderer.AddActor(self.sphereActor)
		if self.make_links:
			self.renderer.AddActor(self.islActor)
			self.renderer.AddActor(self.sglActor)
			self.renderer.AddActor(self.pathActor)

		# white background, makes it easier to
		# put screenshots of animation into papers/presentations
		self.renderer.SetBackground(BACKGROUND_COLOR)

		self.interactor.Initialize()
		print('initialized interactor')

		# set up a timer to call the update function at a max rate
		# of every 7 ms (~144 hz)
		self.interactor.AddObserver('TimerEvent', self.updateAnimation)
		self.interactor.CreateRepeatingTimer(7)
		print('set up timer')

		# start the model
		self.renderWindow.SetSize(512, 512)
		self.renderWindow.Render()
		print('started render')
		self.interactor.Start()
		print('started interactor')

	def makeSatsActor(self, total_satellites, satellite_positions):
		"""
		generate the point cloud to represent satellites

		Parameters
		----------
		total_satellites : int
			number of satellties in the simulation
		satellite_positions : np.array[[('x', int32),('y', int32),('z', int32)]]
			satellite position data, satellite "unique_id" = index number
		"""

		# declare a points & cell array to hold position data
		self.satVtkPts = vtk.vtkPoints()
		self.satVtkVerts = vtk.vtkCellArray()

		# figure out the total number of sats and groundpts in the constillation
		self.totalSats = total_satellites

		# init a array for IDs
		self.satPointIDs = [None] * self.totalSats

		# initialize all the positions
		for i in range(self.totalSats):
			self.satPointIDs[i] = self.satVtkPts.InsertNextPoint(
				satellite_positions[i]['x'],
				satellite_positions[i]['y'],
				satellite_positions[i]['z'])

			self.satVtkVerts.InsertNextCell(1)
			self.satVtkVerts.InsertCellPoint(self.satPointIDs[i])

		# convert points into poly data
		# (because that's what they do in the vtk examples)
		self.satPolyData = vtk.vtkPolyData()
		self.satPolyData.SetPoints(self.satVtkPts)
		self.satPolyData.SetVerts(self.satVtkVerts)

		# create mapper object and connect to the poly data
		self.satsMapper = vtk.vtkPolyDataMapper()
		self.satsMapper.SetInputData(self.satPolyData)

		# create actor, and connect to the mapper
		# (again, its just what you do to make a vtk render pipeline)
		self.satsActor = vtk.vtkActor()
		self.satsActor.SetMapper(self.satsMapper)

		# edit appearance of satellites
		self.satsActor.GetProperty().SetOpacity(SAT_OPACITY)
		self.satsActor.GetProperty().SetColor(SAT_COLOR)
		self.satsActor.GetProperty().SetPointSize(SAT_POINT_SIZE)

	def makeGndPtsActor(self, total_groundpoints, groundpoint_positions):
		"""
		generate the point cloud to represent groundpoints

		Parameters
		----------
		total_groundpoints : int
			number of satellties in the simulation
		groundpoint_positions : np.array[[('x', int32),('y', int32),('z', int32)]]
			ground point position data, satellite "unique_id" = -(index_number-1)
		"""

		# init point and cell arrays
		self.gndVtkPts = vtk.vtkPoints()
		self.gndVtkVerts = vtk.vtkCellArray()

		# figure out the total number of groundpts in the constillation
		self.totalGroundpoints = total_groundpoints

		# init a array for IDs ?
		self.gndPointIDs = [None] * self.totalGroundpoints

		# init positions
		for i in range(self.totalGroundpoints):
			self.gndPointIDs[i] = self.gndVtkPts.InsertNextPoint(
				groundpoint_positions[i]['x'],
				groundpoint_positions[i]['y'],
				groundpoint_positions[i]['z'])

			self.gndVtkVerts.InsertNextCell(1)
			self.gndVtkVerts.InsertCellPoint(self.gndPointIDs[i])

		# more standard pipeline creation...
		self.gndPolyData = vtk.vtkPolyData()
		self.gndPolyData.SetPoints(self.gndVtkPts)
		self.gndPolyData.SetVerts(self.gndVtkVerts)
		self.gndMapper = vtk.vtkPolyDataMapper()
		self.gndMapper.SetInputData(self.gndPolyData)
		self.gndActor = vtk.vtkActor()
		self.gndActor.SetMapper(self.gndMapper)

		# set actor properties
		self.gndActor.GetProperty().SetOpacity(GND_OPACITY)
		self.gndActor.GetProperty().SetColor(GND_COLOR)
		self.gndActor.GetProperty().SetPointSize(GND_POINT_SIZE)

	def makeLinkActors(self):
		"""
		generate the lines to represent links

		source:
		https://vtk.org/Wiki/VTK/Examples/Python/GeometricObjects/Display/PolyLine

		"""

		# grab the arrays of connections
		links = self.model.getArrayOfLinks()
		points = self.model.getArrayOfNodePositions()
		maxSatIdx = self.model.total_sats-1

		# build a vtkPoints object from array
		self.linkPoints = vtk.vtkPoints()
		self.linkPoints.SetNumberOfPoints(len(points))
		for i in range(len(points)):
			self.linkPoints.SetPoint(i, points[i]['x'], points[i]['y'], points[i]['z'])

		# build a cell array to represent connectivity
		self.islLinkLines = vtk.vtkCellArray()
		self.sglLinkLines = vtk.vtkCellArray()
		for i in range(len(links)):
			e1 = links[i]['node_1']
			e2 = links[i]['node_2']
			# must translate link endpoints to point names
			# if endpoint name is positive, we use it directly
			# if negative, idx = maxSatIdx-endpointname
			# **ground endpoints are always node_1**
			if e1 < 0:
				self.sglLinkLines.InsertNextCell(2)
				self.sglLinkLines.InsertCellPoint(maxSatIdx-e1)
				self.sglLinkLines.InsertCellPoint(e2)
			else:
				self.islLinkLines.InsertNextCell(2)
				self.islLinkLines.InsertCellPoint(e1)
				self.islLinkLines.InsertCellPoint(e2)

		self.pathLinkLines = vtk.vtkCellArray()  # init, but do not fill this one

		# #

		self.islPolyData = vtk.vtkPolyData()
		self.islPolyData.SetPoints(self.linkPoints)
		self.islPolyData.SetLines(self.islLinkLines)

		self.sglPolyData = vtk.vtkPolyData()
		self.sglPolyData.SetPoints(self.linkPoints)
		self.sglPolyData.SetLines(self.sglLinkLines)

		self.pathPolyData = vtk.vtkPolyData()
		self.pathPolyData.SetPoints(self.linkPoints)
		self.pathPolyData.SetLines(self.pathLinkLines)

		# #

		self.islMapper = vtk.vtkPolyDataMapper()
		self.islMapper.SetInputData(self.islPolyData)

		self.sglMapper = vtk.vtkPolyDataMapper()
		self.sglMapper.SetInputData(self.sglPolyData)

		self.pathMapper = vtk.vtkPolyDataMapper()
		self.pathMapper.SetInputData(self.pathPolyData)

		# #

		self.islActor = vtk.vtkActor()
		self.islActor.SetMapper(self.islMapper)

		self.sglActor = vtk.vtkActor()
		self.sglActor.SetMapper(self.sglMapper)

		self.pathActor = vtk.vtkActor()
		self.pathActor.SetMapper(self.pathMapper)

		# #

		self.islActor.GetProperty().SetOpacity(ISL_LINK_OPACITY)
		self.islActor.GetProperty().SetColor(ISL_LINK_COLOR)
		self.islActor.GetProperty().SetLineWidth(ISL_LINE_WIDTH)

		self.sglActor.GetProperty().SetOpacity(SGL_LINK_OPACITY)
		self.sglActor.GetProperty().SetColor(SGL_LINK_COLOR)
		self.sglActor.GetProperty().SetLineWidth(SGL_LINE_WIDTH)

		self.pathActor.GetProperty().SetOpacity(PATH_LINK_OPACITY)
		self.pathActor.GetProperty().SetColor(PATH_LINK_COLOR)
		self.pathActor.GetProperty().SetLineWidth(PATH_LINE_WIDTH)

		# #

	def makeEarthActors(self, earth_radius):
		"""
		generate the earth sphere, and the landmass outline

		Parameters
		----------
		earth_radius : int
			radius of the Earth in meters

		"""

		self.earthRadius = earth_radius

		# Create earth map
		# a point cloud that outlines all the earths landmass
		self.earthSource = vtk.vtkEarthSource()
		# draws as an outline of landmass, rather than fill it in
		self.earthSource.OutlineOn()

		# want this to be slightly larger than the sphere it sits on
		# so that it is not occluded by the sphere
		self.earthSource.SetRadius(self.earthRadius * 1.001)

		# controles the resolution of surface data (1 = full resolution)
		self.earthSource.SetOnRatio(1)

		# Create a mapper
		self.earthMapper = vtk.vtkPolyDataMapper()
		self.earthMapper.SetInputConnection(self.earthSource.GetOutputPort())

		# Create an actor
		self.earthActor = vtk.vtkActor()
		self.earthActor.SetMapper(self.earthMapper)

		# set color
		self.earthActor.GetProperty().SetColor(LANDMASS_OUTLINE_COLOR)
		self.earthActor.GetProperty().SetOpacity(EARTH_LAND_OPACITY)

		# make sphere data
		num_pts = EARTH_SPHERE_POINTS
		indices = np.arange(0, num_pts, dtype=float) + 0.5
		phi = np.arccos(1 - 2 * indices / num_pts)
		theta = np.pi * (1 + 5 ** 0.5) * indices
		x = np.cos(theta) * np.sin(phi) * self.earthRadius
		y = np.sin(theta) * np.sin(phi) * self.earthRadius
		z = np.cos(phi) * self.earthRadius

		# x,y,z is coordination of evenly distributed sphere
		# I will try to make poly data use this x,y,z
		points = vtk.vtkPoints()
		for i in range(len(x)):
			points.InsertNextPoint(x[i], y[i], z[i])

		poly = vtk.vtkPolyData()
		poly.SetPoints(points)

		# To create surface of a sphere we need to use Delaunay triangulation
		d3D = vtk.vtkDelaunay3D()
		d3D.SetInputData(poly)  # This generates a 3D mesh

		# We need to extract the surface from the 3D mesh
		dss = vtk.vtkDataSetSurfaceFilter()
		dss.SetInputConnection(d3D.GetOutputPort())
		dss.Update()

		# Now we have our final polydata
		spherePoly = dss.GetOutput()

		# Create a mapper
		sphereMapper = vtk.vtkPolyDataMapper()
		sphereMapper.SetInputData(spherePoly)

		# Create an actor
		self.sphereActor = vtk.vtkActor()
		self.sphereActor.SetMapper(sphereMapper)

		# set color
		self.sphereActor.GetProperty().SetColor(EARTH_BASE_COLOR)
		self.sphereActor.GetProperty().SetOpacity(EARTH_OPACITY)
