######################################################################
#                                                                    
# Part of SILLEO-SCNS, Core functions for satellite and network simulation
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

# gotta have numpy
import numpy as np

# used to calculate kepler 2 body orbits
from PyAstronomy import pyasl

import math

import networkx as nx

# try to import numba funcs
try:
	import numba
	USING_NUMBA = True
except ModuleNotFoundError:
	USING_NUMBA = False
	print("you probably do not have numba installed...")
	print("reverting to non-numba mode")

PRINT_LEVEL = 0           # the higher the number, the more stuff printed


def sdout(level=0, *args):
	"""a print function; chooses to print based on PRINT_LEVEL

	Parameters
	----------
	level : int
		print_level, higher = more printing
	args :
		the stuff to print

	Returns
	-------
	None
	"""

	if level <= PRINT_LEVEL:
		toPrint = ''
		for arg in args:
			toPrint += str(arg)
		print(toPrint)
	return None


# the mean radius of the earth in meters according to wikipedia
EARTH_RADIUS = 6371000

# earth's z axis (eg a vector in the positive z direction)
EARTH_ROTATION_AXIS = [0, 0, 1]

# number of seconds per earth rotation (day)
SECONDS_PER_DAY = 86400

# according to wikipedia
STD_GRAVITATIONAL_PARAMATER_EARTH = 3.986004418e14

# how big to initialize the ground point array...
NUM_GROUND_POINTS = 0

# The numpy data type used to store satellite data
# note  use of int16 for names: max number of satellites = (2^15)-1
# note use of int32 for position: max pos value = (2^31)-1 meters
#    (this is 5.5 times the distance to the moon
#     and should be fine for earth orbit simulation)
# Note that fields can be accessed by their name: array[idx]['ID']
SATELLITE_DTYPE = np.dtype([
	('ID', np.int16),             # ID number, unique, = array index
	('plane_number', np.int16),   # which orbital plane is the satellite in?
	('offset_number', np.int16),  # What satellite withen the plane?
	('time_offset', np.float32),  # time offset for kepler ellipse solver
	('x', np.int32),              # x position in meters
	('y', np.int32),              # y position in meters
	('z', np.int32)])             # z position in meters


# The numpy data type used to store ground point data
# ground points have negative unique IDs
# positions are always calculated from the initial position
# to keep rounding error from compounding
GROUNDPOINT_DTYPE = np.dtype([
	('ID', np.int16),      # ID number, unique, = array index
	('init_x', np.int32),  # initial x position in meters
	('init_y', np.int32),  # initial y position in meters
	('init_z', np.int32),  # initial z position in meters
	('x', np.int32),       # x position in meters
	('y', np.int32),       # y position in meters
	('z', np.int32)])      # z position in meters


# The numpy data type used to store link data
# link array size may have to be adjusted
# each index is 8 bytes
LINK_DTYPE = np.dtype([
	('node_1', np.int16),     # an endpoint of the link
	('node_2', np.int16),     # the other endpoint of the link
	('distance', np.int32)])  # distance of the link in meters


LINK_ARRAY_SIZE = 10000000  # 10 million indices = 80 megabyte array (huge)

###############################################################################
# const class


class Constellation():
	"""
	A class used to contain and manage a satellite constillation

	...

	Attributes
	----------
	number_of_planes : int
		the number of planes in the constillation
	nodes_per_plane : int
		the number of satellites per plane
	total_sats : int
		the total number of nodes(satellites) in constillation
	ground_node_counter : int
		always negative, countdown, used to keep track of ground node IDs
	inclination : float
		the inclination of all planes in constillation
	semi_major_axis : float
		semi major axis of the orbits (radius, if orbits circular)
	period : int
		the period of the orbits in seconds
	eccentricity : float
		the eccentricity of the orbits; range = 0.0 - 1.0
	satellites_array : SATELLITE_DTYPE
		numpy array of satellite_dtype, contains satellite data
	raan_offsets : List[float]
		list of floats, keeps track of all the ascending node offsets in degrees
	plane_solvers : List[ke_solver]
		contains the PyAstronomy Kepler Ellipse solver for each orbital plane
	time_offsets : List[float]
		contains the time offsets for satellties withen a plane
	current_time : int
		keeps track of the current simulation time

	Methods
	-------
	initSatelliteArray(sat_array=None)
		fills the sat array with initial values for time=0
	getArrayOfNodePositions()
		returns a a slice of the constillation array, containing only position values
	setConstillationTime(time=0.0)
		updates all satellites and ground stations positions to reflect the new time
	"""

	def __init__(
			self,
			planes=1,
			nodes_per_plane=4,
			inclination=0,
			semi_major_axis=6372000,
			ecc=0.0,
			minCommunicationsAltitude=100000,
			minSatElevation=40,
			linkingMethod='SPARSE',
			arcOfAscendingNodes=360.0):
		"""
		Parameters
		----------
		planes : int
			the number of planes in the constillation
		nodes_per_plane : int
			the number of satellites per plane
		inclination : float
			the inclination of all planes in constillation
		semi_major_axis : float
			semi major axis of the orbits (radius, if orbits circular)
		ecc : float
			the eccentricity of the orbits; range = 0.0 - 1.0
		minCommunicationsAltitude : int32
			The minimum altitude that inter satellite links must pass
			above the Earth's surface.
		minSatElevation : int
			The minimum angle of elevation in degrees above the horizon a satellite
			needs to have for a ground station to communicate with it.
		linkingMethod : string
			The current linking method used by the constillation
			currently only used for generating GML files.
		arcOfAscendingNodes : float
			The angle of arc (in degrees) that the ascending nodes of all the
			orbital planes is evenly spaced along. Ex, seting this to 180 results
			in a Pi constellation like Iridium
		"""

		self.number_of_planes = planes
		self.nodes_per_plane = nodes_per_plane
		self.total_sats = planes*nodes_per_plane
		self.ground_node_counter = 0
		self.inclination = inclination
		self.semi_major_axis = semi_major_axis
		self.period = self.calculateOrbitPeriod(semi_major_axis=self.semi_major_axis)
		self.eccentricity = ecc
		self.current_time = 0
		self.number_of_isl_links = 0
		self.number_of_gnd_links = 0
		self.total_links = 0
		self.link_array_size = LINK_ARRAY_SIZE
		self.min_communications_altitude = 100000
		self.min_sat_elevation = 40
		self.linking_method = 'SPARSE'
		self.G = None

		# this is not written to zero, because it has it's own init
		# function a a few lines down: initSatelliteArray()
		self.satellites_array = np.empty(self.total_sats, dtype=SATELLITE_DTYPE)

		# declare an empty ground
		self.groundpoints_array = np.zeros(NUM_GROUND_POINTS,
		                                   dtype=GROUNDPOINT_DTYPE)

		# declare an empty link array
		self.link_array = np.zeros(self.link_array_size, dtype=LINK_DTYPE)

		# figure out how many degrees to space right ascending nodes of the planes
		self.raan_offsets = [(arcOfAscendingNodes / self.number_of_planes)*i for i in
		                     range(0, self.number_of_planes)]

		# generate a list with a kepler ellipse solver object for each plane
		self.plane_solvers = []
		for raan in self.raan_offsets:
			self.plane_solvers.append(pyasl.KeplerEllipse(
				per=self.period,         # how long the orbit takes in seconds
				a=self.semi_major_axis,  # if circular orbit, this is same as radius
				e=self.eccentricity,     # generally close to 0 for leo constillations
				Omega=raan,              # right ascention of the ascending node
				w=0.0,                   # initial time offset / mean anamoly
				i=self.inclination))     # orbit inclination

		# figure out the time offsets for nodes withen a plane
		self.time_offsets = [(self.period/nodes_per_plane)*i for i in
		                     range(0, nodes_per_plane)]

		# initialize the satellite array
		self.initSatelliteArray()

	def initSatelliteArray(self):
		"""initializes the satellite array with positions at time zero

		Parameters
		----------
		sat_array :
			the satellite array object, modified in place

		"""

		# we offset each plane by a small amount, so they do not 'collide'
		# this little algorithm comes up with a list of offset values
		phase_offset = 0
		phase_offset_increment = ((self.period / self.nodes_per_plane) /
		                          self.number_of_planes)
		temp = []
		toggle = False
		# this loop results puts thing in an array in this order:
		# [...8,6,4,2,0,1,3,5,7...]
		# so that the offsets in adjacent planes are similar
		# basically do not want the max and min offset in two adjcent planes
		for i in range(self.number_of_planes):
			if toggle:
				temp.append(phase_offset)
			else:
				temp.insert(0, phase_offset)
				# temp.append(phase_offset)
			toggle = not toggle
			phase_offset = phase_offset + phase_offset_increment

		phase_offsets = temp

		# randomly shuffle the list...
		# random.shuffle(temp)

		# arrange by even Odd
		# phase_offsets = []
		# for i in range(int(len(temp)/2)+1):
		# 	phase_offsets.append(temp[i])
		# 	i_2 = i + int(len(temp)/2)+1
		# 	if i_2 < (len(temp)):
		# 		phase_offsets.append(temp[i_2])

		# loop through all satellites
		for plane in range(0, self.number_of_planes):
			for node in range(0, self.nodes_per_plane):

				# calculate the KE solver time offset
				offset = (self.time_offsets[node] + phase_offsets[plane])

				# calculate the unique ID of the node (same as array index)
				unique_id = (plane*self.nodes_per_plane) + node

				# calculate initial position
				init_pos = self.plane_solvers[plane].xyzPos(offset)

				# update satellties array
				self.satellites_array[unique_id]['ID'] = np.int16(unique_id)
				self.satellites_array[unique_id]['plane_number'] = np.int16(plane)
				self.satellites_array[unique_id]['offset_number'] = np.int16(node)
				self.satellites_array[unique_id]['time_offset'] = np.float32(offset)
				self.satellites_array[unique_id]['x'] = np.int32(init_pos[0])
				self.satellites_array[unique_id]['y'] = np.int32(init_pos[1])
				self.satellites_array[unique_id]['z'] = np.int32(init_pos[2])

	def getArrayOfNodePositions(self):
		"""copies a sub array of only position data from
		satellite AND groundpoint arrays

		Returns
		-------
		positions : np array
			a copied sub array of the satellite array, that only contains positions data
		"""

		sat_positions = np.copy(self.satellites_array[['x', 'y', 'z']])
		ground_positions = np.copy(self.groundpoints_array[['x', 'y', 'z']])

		# combine sat and ground positions into a 'positions' array
		positions = np.append(sat_positions, ground_positions)
		return positions

	def getArrayOfSatPositions(self):
		"""copies a sub array of only position data from
		satellite array

		Returns
		-------
		sat_positions : np array
			a copied sub array of the satellite array, that only contains positions data
		"""

		sat_positions = np.copy(self.satellites_array[['x', 'y', 'z']])

		return sat_positions

	def getArrayOfGndPositions(self):
		"""copies a sub array of only position data from
		 groundpoint array

		Returns
		-------
		ground_positions : np array
			a copied sub array of the ground point array, that only contains positions
		"""

		ground_positions = np.copy(self.groundpoints_array[['x', 'y', 'z']])

		return ground_positions

	def getArrayOfLinks(self):
		"""copies a sub array of link data

		Returns
		-------
		links : np array
			contains all links
		"""
		total_links = self.total_links
		links = np.copy(self.link_array[:total_links-1])

		return links

	def setConstillationTime(self, time=0.0):
		"""updates all position and link data to specified time

		Parameters
		----------
		time : float
			simulation time to set to in seconds

		Returns
		-------
		None
		"""

		# cast time to an int
		self.current_time = int(time)

		# update all the satellite positions
		for sat_id in range(self.satellites_array.size):
			plane = self.satellites_array[sat_id]['plane_number']
			offset = self.satellites_array[sat_id]['time_offset']
			pos = self.plane_solvers[plane].xyzPos(self.current_time + offset)
			self.satellites_array[sat_id]['x'] = np.int32(pos[0])
			self.satellites_array[sat_id]['y'] = np.int32(pos[1])
			self.satellites_array[sat_id]['z'] = np.int32(pos[2])

		# update all the ground point positions
		if self.current_time == 0 or self.current_time % SECONDS_PER_DAY == 0:
			degrees_to_rotate = 0
		else:
			degrees_to_rotate = 360.0/(SECONDS_PER_DAY /
			                           (self.current_time % SECONDS_PER_DAY))

		rotation_matrix = self.getRotationMatrix(EARTH_ROTATION_AXIS,
		                                         degrees_to_rotate)

		for gnd_pt in range(self.groundpoints_array.size):
			initial_pos = self.groundpoints_array[gnd_pt][
				['init_x', 'init_y', 'init_z']]
			initial_pos = [initial_pos[0], initial_pos[1], initial_pos[2]]
			new_pos = np.dot(rotation_matrix, initial_pos)
			self.groundpoints_array[gnd_pt]['x'] = new_pos[0]
			self.groundpoints_array[gnd_pt]['y'] = new_pos[1]
			self.groundpoints_array[gnd_pt]['z'] = new_pos[2]

		return None

	def generateNetworkGraph(self, city_names):
		""" Makes a NetworkX graph of the network at the current time.

		"""
		self.G = nx.Graph(
			numPlanes=str(self.number_of_planes),
			numNodesPerPlane=str(self.nodes_per_plane),
			planeInclination=str(self.inclination),
			semiMajorAxisMeters=str(self.semi_major_axis),
			minCommunicationsAltitudeMeters=str(self.min_communications_altitude),
			minSatElevationDegrees=str(self.min_sat_elevation),
			simulationTime=str(self.current_time),
			connectionStrategy=self.linking_method)

		# now add in sats
		# remember, for sats, the array index = sat ID
		for sat_idx in range(self.total_sats):
			self.G.add_node(
				str(self.satellites_array[sat_idx]['ID']),
				planeNumber=str(self.satellites_array[sat_idx]['plane_number']),
				offsetNumber=str(self.satellites_array[sat_idx]['offset_number']))

		# now add all the ground nodes
		# gnd pts have negative ID numbers
		for gnd_idx in range((-self.ground_node_counter)):
			self.G.add_node(str(self.groundpoints_array[gnd_idx]['ID']),
				placeName=city_names[gnd_idx])

		# and finally the links (edges in nx terms)
		for lnk_idx in range(self.total_links):
			self.G.add_edge(
				str(self.link_array[lnk_idx]['node_1']),
				str(self.link_array[lnk_idx]['node_2']),
				distance=int(self.link_array[lnk_idx]['distance']))

	def exportGMLFile(self, filename):
		""" Exports a GML file of the current graph (self.G)

		"""
		nx.write_gml(self.G, filename)

	def calculateOrbitPeriod(self, semi_major_axis=0.0):
		""" calculates the period of a orbit for Earth

		Parameters
		----------
		semi_major_axis : float
			semi major axis of the orbit in meters

		Returns
		-------
		Period : int
			the period of the orbit in seconds (rounded to whole seconds)
		"""

		tmp = math.pow(semi_major_axis, 3) / STD_GRAVITATIONAL_PARAMATER_EARTH
		return int(2.0 * math.pi * math.sqrt(tmp))

	def addGroundPoint(self, latitude, longitude, altitude=100.0):
		""" adds a ground point at given coordinates, assumes earth is perfect sphere

		Parameters
		----------
		latitude : float
			latitude of ground point (in degrees)
		longitude : float
			longitude of ground point (in degrees)
		altitude : float
			altitude of point in meters (0 = earth surface)

		Returns
		-------
		unique_id : int
			the ID value assigned to ground point (will be < 0)
		"""

		# must convert the lat/long/alt to cartesian coordinates
		radius = EARTH_RADIUS + altitude
		init_pos = [0, 0, 0]
		latitude = math.radians(latitude)
		longitude = math.radians(longitude)
		init_pos[0] = radius * math.cos(latitude) * math.cos(longitude)
		init_pos[1] = radius * math.cos(latitude) * math.sin(longitude)
		init_pos[2] = radius * math.sin(latitude)

		# be sure to decrement this for the next ground point
		self.ground_node_counter = self.ground_node_counter - 1
		unique_id = self.ground_node_counter

		# if simulation time is not 0, figure out current position
		if self.current_time == 0 or self.current_time % SECONDS_PER_DAY == 0:
			degrees_to_rotate = 0
			pos = init_pos

		else:
			degrees_to_rotate = 360.0/(SECONDS_PER_DAY /
			                           (self.current_time % SECONDS_PER_DAY))

			rotation_matrix = self.getRotationMatrix(EARTH_ROTATION_AXIS,
			                                         degrees_to_rotate)

			pos = np.dot(rotation_matrix, init_pos)

		# add the new ground point to array
		# yes, append means a full array copy every time,
		# but this should be a very small array,
		# and ground points are probably only added once
		# at the begining of the simulation
		temp = np.zeros(1, dtype=GROUNDPOINT_DTYPE)
		temp[0]['ID'] = np.int16(unique_id)
		temp[0]['init_x'] = np.int32(init_pos[0])
		temp[0]['init_y'] = np.int32(init_pos[1])
		temp[0]['init_z'] = np.int32(init_pos[2])
		temp[0]['x'] = np.int32(pos[0])
		temp[0]['y'] = np.int32(pos[1])
		temp[0]['z'] = np.int32(pos[2])

		self.groundpoints_array = np.append(self.groundpoints_array, temp)

		return unique_id

	def getRotationMatrix(self, axis, degrees):
		"""
		Return the rotation matrix associated with counterclockwise rotation about
		the given axis by theta radians.

		Parameters
		----------
		axis : list[x, y, z]
			a vector defining the rotaion axis
		degrees : float
			The number of degrees to rotate

		"""

		theta = math.radians(degrees)
		axis = np.asarray(axis)
		axis = axis / math.sqrt(np.dot(axis, axis))
		a = math.cos(theta / 2.0)
		b, c, d = -axis * math.sin(theta / 2.0)
		aa, bb, cc, dd = a * a, b * b, c * c, d * d
		bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
		return np.array([
			[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
			[2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
			[2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

	def calculateMaxISLDistance(self, min_communication_altitude):
		"""
		ues some trig to calculate the max coms range between satellites
		based on some minium communications altitude

		Parameters
		----------
		min_communication_altitude : int
			min coms altitude in meters, referenced from Earth's surface

		Returns
		-------
		max distance : int
			max distance in meters

		"""

		c = EARTH_RADIUS + min_communication_altitude
		b = self.semi_major_axis
		B = math.radians(90)
		C = math.asin((c * math.sin(B)) / b)
		A = math.radians(180) - B - C
		a = (b * math.sin(A)) / math.sin(B)
		return int(a * 2)

	def calculateMaxSpaceToGndDistance(self, min_elevation):
		"""
		Return max satellite to ground coms distance

		Uses some trig to calculate the max space to ground communications
		distance given a field of view for groundstations defined by an
		minimum elevation angle above the horizon.
		Uses a circle & line segment intercept calculation.

		Parameters
		----------
		min_elevation : int
			min elevation in degrees, range: 0<val<90

		Returns
		-------
		max distance : int
			max coms distance in meters

		"""

		# TODO
		# make a drawing explaining this

		full_line = False
		tangent_tol = 1e-9

		# point 1 of line segment, representing groundstation
		p1x, p1y = (0, EARTH_RADIUS)

		# point 2 of line segment, representing really far point
		# at min_elevation slope from point 1
		slope = math.tan(math.radians(min_elevation))
		run = 384748000  # meters, sma of moon
		rise = slope * run + EARTH_RADIUS
		p2x, p2y = (run, rise)

		# center of orbit circle = earth center
		# radius = orbit radius
		cx, cy = (0, 0)
		circle_radius = self.semi_major_axis

		(x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
		dx, dy = (x2 - x1), (y2 - y1)
		dr = (dx ** 2 + dy ** 2)**.5
		big_d = x1 * y2 - x2 * y1
		discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

		if discriminant < 0:  # No intersection between circle and line
			print("ERROR! problem with calculateMaxSpaceToGndDistance, no intersection")
			return 0
		else:  # There may be 0, 1, or 2 intersections with the segment
			intersections = [
				(cx+(big_d*dy+sign*(-1 if dy < 0 else 1)*dx*discriminant**.5)/dr**2,
				 cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
				for sign in ((1, -1) if dy < 0 else (-1, 1))]

			# This makes sure the order along the segment is correct
			if not full_line:
				# Filter out intersections that do not fall within the segment
				fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy)
				                          else (yi - p1y) / dy for xi, yi in intersections]

				intersections = [pt for pt, frac in
				                 zip(intersections, fraction_along_segment)
				                 if 0 <= frac <= 1]

			if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
				# If line is tangent to circle, return just one point
				print("ERROR!, got 2 intersections, expecting 1")
				return 0
			else:
				ints_lst = intersections

		# assuming 2 intersections were found...
		for i in ints_lst:
			if i[1] < 0:
				continue
			else:
				# calculate dist to this intersection
				d = math.sqrt(
					math.pow(i[0]-p1x, 2) +
					math.pow(i[1]-p1y, 2)
				)
				return int(d)

	def calculateIdealLinks(self, max_isl_range, max_stg_range):
		"""
		figure out all possible inter-satellite links
		for each satellite, with a distance less than max_coms_range

		Parameters
		----------
		max_coms_range : int
			the max coms range in meters

		"""

		if USING_NUMBA is True:
			temp = self.numba_calculateIdealLinks(
				max_isl_range,
				max_stg_range,
				self.total_sats,
				self.satellites_array,
				self.link_array,
				self.groundpoints_array,
				self.ground_node_counter,
				self.link_array_size)
			if temp is not None:
				self.number_of_isl_links = temp[0]
				self.number_of_gnd_links = temp[1]
				self.total_links = temp[2]

		else:

			link_idx = 0

			# add the ISL links
			for sat_idx_a in range(self.total_sats - 1):
				# copy the position of sat a
				sat_a_pos = [
					self.satellites_array[sat_idx_a]['x'],
					self.satellites_array[sat_idx_a]['y'],
					self.satellites_array[sat_idx_a]['z']
				]
				for sat_idx_b in range(sat_idx_a + 1, self.total_sats):
					# calculate distance from a to b
					d = int(math.sqrt(
						math.pow(self.satellites_array[sat_idx_b]['x'] - sat_a_pos[0], 2) +
						math.pow(self.satellites_array[sat_idx_b]['y'] - sat_a_pos[1], 2) +
						math.pow(self.satellites_array[sat_idx_b]['z'] - sat_a_pos[2], 2)))
					# deicide if link is valid or not
					if d < max_isl_range:
						if link_idx < self.link_array_size - 1:
							self.link_array[link_idx]['node_1'] = np.int16(sat_idx_a)
							self.link_array[link_idx]['node_2'] = np.int16(sat_idx_b)
							self.link_array[link_idx]['distance'] = np.int32(d)
							link_idx = link_idx + 1
						else:
							print('ERROR! ran out of room in the link array')
							return

			self.number_of_isl_links = link_idx

			# add the StG links
			for gnd_idx in range(-self.ground_node_counter):
				gnd_pos = [
					self.groundpoints_array[gnd_idx]['x'],
					self.groundpoints_array[gnd_idx]['y'],
					self.groundpoints_array[gnd_idx]['z']
				]
				for sat_idx in range(self.total_sats):
					# calculate distance
					d = int(math.sqrt(
						math.pow(self.satellites_array[sat_idx]['x'] - gnd_pos[0], 2) +
						math.pow(self.satellites_array[sat_idx]['y'] - gnd_pos[1], 2) +
						math.pow(self.satellites_array[sat_idx]['z'] - gnd_pos[2], 2)))
					# deicide if link is valid or not
					if d < max_stg_range:
						if link_idx < self.link_array_size - 1:
							gnd_id = self.groundpoints_array[gnd_idx]['ID']
							sat_id = self.satellites_array[sat_idx]['ID']
							self.link_array[link_idx]['node_1'] = gnd_id
							self.link_array[link_idx]['node_2'] = sat_id
							self.link_array[link_idx]['distance'] = np.int32(d)
							link_idx = link_idx + 1
						else:
							print('ERROR! ran out of room in the link array')
							return

				self.number_of_gnd_links = link_idx - self.number_of_isl_links
				self.total_links = link_idx

	@staticmethod
	@numba.jit(nopython=True)
	def numba_calculateIdealLinks(
			max_isl_range,
			max_stg_range,
			total_sats,
			satellites_array,
			link_array,
			groundpoints_array,
			ground_node_counter,
			link_array_size):
		"""
		figure out all possible inter-satellite links
		for each satellite, with a distance less than max_coms_range

		Parameters
		----------
		max_coms_range : int
			the max coms range in meters

		"""

		link_idx = 0

		# add the ISL links
		for sat_idx_a in range(total_sats - 1):
			# copy the position of sat a
			sat_a_pos = [
				satellites_array[sat_idx_a]['x'],
				satellites_array[sat_idx_a]['y'],
				satellites_array[sat_idx_a]['z']
			]
			for sat_idx_b in range(sat_idx_a + 1, total_sats):
				# calculate distance from a to b
				d = int(math.sqrt(
					math.pow(satellites_array[sat_idx_b]['x'] - sat_a_pos[0], 2) +
					math.pow(satellites_array[sat_idx_b]['y'] - sat_a_pos[1], 2) +
					math.pow(satellites_array[sat_idx_b]['z'] - sat_a_pos[2], 2)))
				# deicide if link is valid or not
				if d < max_isl_range:
					if link_idx < link_array_size - 1:
						link_array[link_idx]['node_1'] = np.int16(sat_idx_a)
						link_array[link_idx]['node_2'] = np.int16(sat_idx_b)
						link_array[link_idx]['distance'] = np.int32(d)
						link_idx = link_idx + 1
					else:
						# print('ERROR! ran out of room in the link array')
						return

		number_of_isl_links = link_idx

		# add the StG links
		for gnd_idx in range(0, -(ground_node_counter)):
			gnd_pos = [
				groundpoints_array[gnd_idx]['x'],
				groundpoints_array[gnd_idx]['y'],
				groundpoints_array[gnd_idx]['z']
			]
			for sat_idx in range(total_sats):
				# calculate distance
				d = int(math.sqrt(
					math.pow(satellites_array[sat_idx]['x'] - gnd_pos[0], 2) +
					math.pow(satellites_array[sat_idx]['y'] - gnd_pos[1], 2) +
					math.pow(satellites_array[sat_idx]['z'] - gnd_pos[2], 2)))
				# deicide if link is valid or not
				if d < max_stg_range:
					d = np.int32(d)
					if link_idx < link_array_size - 1:
						gnd_id = groundpoints_array[gnd_idx]['ID']
						sat_id = satellites_array[sat_idx]['ID']
						link_array[link_idx]['node_1'] = gnd_id
						link_array[link_idx]['node_2'] = sat_id
						link_array[link_idx]['distance'] = d
						link_idx = link_idx + 1
					else:
						print('ERROR! ran out of room in the link array')
						return

		number_of_gnd_links = link_idx - number_of_isl_links
		total_links = link_idx

		return [number_of_isl_links, number_of_gnd_links, total_links]

	def calculatePlusGridLinks(
			self,
			max_stg_range,
			max_isl_range=(2**31)-1,
			initialize=False,
			crosslink_interpolation=1):
		"""
		connect satellites in a +grid network

		Parameters
		----------
		max_stg_range : int
			the max space-ground coms range
		initialize : bool
			Because PlusGrid ISL are static, they only need to be generated once,
			If initialize=False, only update link distances, do not regererate
		crosslink_interpolation : int
			This value is used to make only 1 out of every crosslink_interpolation
			satellites able to have crosslinks. For example, with a interpolation
			value of '2', only every other satellite will have crosslinks, the rest
			will have only intra-plane links

		"""

		# TODO:split this into two functions:
		# initialize_plus_grid_links()
		# 	just inits plus grid links, does not calculate distances
		# update_link_distances()
		# 	just goes through existing links and recalculates distances
		# 	same as calling this with initialize=false

		if initialize:
			self.number_of_isl_links = 0

		if USING_NUMBA is True:
			temp = self.numba_calculatePlusGridLinks(
				max_stg_range,
				self.total_sats,
				self.satellites_array,
				self.link_array,
				self.groundpoints_array,
				self.ground_node_counter,
				self.link_array_size,
				self.number_of_planes,
				self.nodes_per_plane,
				number_of_isl_links=self.number_of_isl_links,
				initialize=initialize,
				crosslink_interpolation=crosslink_interpolation,
				max_isl_range=max_isl_range)
			if temp is not None:
				self.number_of_isl_links = temp[0]
				self.number_of_gnd_links = temp[1]
				self.total_links = temp[2]

		else:

			if initialize:

				link_idx = 0

				# add the intra-plane links
				for plane in range(self.number_of_planes):
					for node in range(self.nodes_per_plane):
						node_1 = node + (plane * self.nodes_per_plane)
						if node == self.nodes_per_plane - 1:
							node_2 = plane * self.nodes_per_plane
						else:
							node_2 = node + (plane * self.nodes_per_plane) + 1

						if link_idx < self.link_array_size - 1:
							self.link_array[link_idx]['node_1'] = np.int16(node_1)
							self.link_array[link_idx]['node_2'] = np.int16(node_2)
							link_idx = link_idx + 1
						else:
							print('ERROR! ran out of room in the link array for intra-plane links')
							return
				# add the cross-plane links
				for plane in range(self.number_of_planes):
					if plane == self.number_of_planes - 1:
						plane2 = 0
					else:
						plane2 = plane + 1
					for node in range(self.nodes_per_plane):
						node_1 = node + (plane * self.nodes_per_plane)
						node_2 = node + (plane2 * self.nodes_per_plane)
						if link_idx < self.link_array_size - 1:
							if (node_1 + 1) % crosslink_interpolation == 0:
								self.link_array[link_idx]['node_1'] = np.int16(node_1)
								self.link_array[link_idx]['node_2'] = np.int16(node_2)
								link_idx = link_idx + 1
						else:
							print('ERROR! ran out of room in the link array for cross-plane links')
							return

				self.number_of_isl_links = link_idx

			link_idx = self.number_of_isl_links

			# update ISL link distances
			for isl_idx in range(self.number_of_isl_links):
				sat_1 = self.link_array[isl_idx]['node_1']
				sat_2 = self.link_array[isl_idx]['node_2']
				d = int(math.sqrt(
					math.pow(self.satellites_array[sat_1]['x'] -
									self.satellites_array[sat_2]['x'], 2) +
					math.pow(self.satellites_array[sat_1]['y'] -
									self.satellites_array[sat_2]['y'], 2) +
					math.pow(self.satellites_array[sat_1]['z'] -
									self.satellites_array[sat_2]['z'], 2)))
				if d > max_isl_range:
					self.link_array[isl_idx]['node_1'] = np.int16(0)
					self.link_array[isl_idx]['node_2'] = np.int16(0)
					self.link_array[isl_idx]['distance'] = np.int32(0)
				else:
					self.link_array[isl_idx]['distance'] = np.int32(d)

			# add the StG links
			for gnd_idx in range(-self.ground_node_counter):
				gnd_pos = [
					self.groundpoints_array[gnd_idx]['x'],
					self.groundpoints_array[gnd_idx]['y'],
					self.groundpoints_array[gnd_idx]['z']
				]

				for sat_idx in range(self.total_sats):
					# calculate distance
					d = int(math.sqrt(
						math.pow(self.satellites_array[sat_idx]['x'] - gnd_pos[0], 2) +
						math.pow(self.satellites_array[sat_idx]['y'] - gnd_pos[1], 2) +
						math.pow(self.satellites_array[sat_idx]['z'] - gnd_pos[2], 2)))

					# deicide if link is valid or not
					if d < max_stg_range:
						if link_idx < self.link_array_size - 1:
							gnd_id = self.groundpoints_array[gnd_idx]['ID']
							sat_id = self.satellites_array[sat_idx]['ID']
							self.link_array[link_idx]['node_1'] = gnd_id
							self.link_array[link_idx]['node_2'] = sat_id
							self.link_array[link_idx]['distance'] = np.int32(d)
							link_idx = link_idx + 1
						else:
							print('ERROR! ran out of room in the link array')
							return

				self.number_of_gnd_links = link_idx - self.number_of_isl_links
				self.total_links = link_idx

	@staticmethod
	@numba.jit(nopython=True)
	def numba_calculatePlusGridLinks(
			max_stg_range,
			total_sats,
			satellites_array,
			link_array,
			groundpoints_array,
			ground_node_counter,
			link_array_size,
			number_of_planes,
			nodes_per_plane,
			number_of_isl_links,
			initialize=False,
			crosslink_interpolation=1,
			max_isl_range=(2**31)-1):
		"""
		figure out all possible inter-satellite links
		for each satellite, with a distance less than max_coms_range

		Parameters
		----------
		max_stg_range : int
			the max space-ground coms range
		initialize : bool
			Because PlusGrid ISL are static, they only need to be generated once,
			If initialize=False, only update link distances, do not regererate
		crosslink_interpolation : int
			This value is used to make only 1 out of every crosslink_interpolation
			satellites able to have crosslinks. For example, with a interpolation
			value of '2', only every other satellite will have crosslinks, the rest
			will have only intra-plane links

		"""

		if initialize:

			link_idx = 0

			# add the intra-plane links
			for plane in range(number_of_planes):
				for node in range(nodes_per_plane):
					node_1 = node + (plane * nodes_per_plane)
					if node == nodes_per_plane - 1:
						node_2 = plane * nodes_per_plane
					else:
						node_2 = node + (plane * nodes_per_plane) + 1

					if link_idx < link_array_size - 1:
						link_array[link_idx]['node_1'] = np.int16(node_1)
						link_array[link_idx]['node_2'] = np.int16(node_2)
						link_idx = link_idx + 1
					else:
						print('ERROR! ran out of room in the link array for intra-plane links')
						return
			# add the cross-plane links
			for plane in range(number_of_planes):
				if plane == number_of_planes - 1:
					plane2 = 0
				else:
					plane2 = plane + 1
				for node in range(nodes_per_plane):
					node_1 = node + (plane * nodes_per_plane)
					node_2 = node + (plane2 * nodes_per_plane)
					if link_idx < link_array_size - 1:
						if (node_1 + 1) % crosslink_interpolation == 0:
							link_array[link_idx]['node_1'] = np.int16(node_1)
							link_array[link_idx]['node_2'] = np.int16(node_2)
							link_idx = link_idx + 1
					else:
						print('ERROR! ran out of room in the link array for cross-plane links')
						return

			number_of_isl_links = link_idx

		link_idx = number_of_isl_links

		# update ISL link distances
		for isl_idx in range(number_of_isl_links):
			sat_1 = link_array[isl_idx]['node_1']
			sat_2 = link_array[isl_idx]['node_2']
			d = int(math.sqrt(
				math.pow(satellites_array[sat_1]['x'] - satellites_array[sat_2]['x'], 2) +
				math.pow(satellites_array[sat_1]['y'] - satellites_array[sat_2]['y'], 2) +
				math.pow(satellites_array[sat_1]['z'] - satellites_array[sat_2]['z'], 2)))
			if d > max_isl_range:
				link_array[isl_idx]['node_1'] = np.int16(0)
				link_array[isl_idx]['node_2'] = np.int16(0)
				link_array[isl_idx]['distance'] = np.int32(0)
			else:
				link_array[isl_idx]['distance'] = np.int32(d)

		# add the StG links
		for gnd_idx in range(-ground_node_counter):
			gnd_pos = [
				groundpoints_array[gnd_idx]['x'],
				groundpoints_array[gnd_idx]['y'],
				groundpoints_array[gnd_idx]['z']
			]

			for sat_idx in range(total_sats):
				# calculate distance
				d = int(math.sqrt(
					math.pow(satellites_array[sat_idx]['x'] - gnd_pos[0], 2) +
					math.pow(satellites_array[sat_idx]['y'] - gnd_pos[1], 2) +
					math.pow(satellites_array[sat_idx]['z'] - gnd_pos[2], 2)))

				# deicide if link is valid or not
				if d < max_stg_range:
					if link_idx < link_array_size - 1:
						gnd_id = groundpoints_array[gnd_idx]['ID']
						sat_id = satellites_array[sat_idx]['ID']
						link_array[link_idx]['node_1'] = gnd_id
						link_array[link_idx]['node_2'] = sat_id
						link_array[link_idx]['distance'] = np.int32(d)
						link_idx = link_idx + 1
					else:
						print('ERROR! ran out of room in the link array')
						return

		number_of_gnd_links = link_idx - number_of_isl_links
		total_links = link_idx
		return [number_of_isl_links, number_of_gnd_links, total_links]

	def import_links_from_gml_data(self, links):
		""" Takes a links array, and fills internal links_array with them.

		Paramaters
		----------
		links : array[ [(int : node_1_ID), (int : node_2_ID)], ... ]
			An array where each index is a pair of endpoint IDs describing a link.

		"""

		link_idx = 0

		# add the inter satellite links
		for idx in range(len(links)):
			node_1 = links[idx][0]
			node_2 = links[idx][1]
			# if both node IDs are positive (both satellites)
			# we add them here
			if (node_1 >= 0) and (node_2 >= 0):
				if link_idx < self.link_array_size - 1:
					self.link_array[link_idx]['node_1'] = np.int16(node_1)
					self.link_array[link_idx]['node_2'] = np.int16(node_2)
					link_idx = link_idx + 1
				else:
					print('ERROR! ran out of room in the link array for intra-plane links')
					return None

		self.number_of_isl_links = link_idx

		# no we go back over the links array, and add
		# all the links involving groundpoints (negative IDs)
		for idx in range(len(links)):
			node_1 = links[idx][0]
			node_2 = links[idx][1]
			# if one of the IDs is negative, it is a ground point
			if (node_1 < 0) or (node_2 < 0):
				if link_idx < self.link_array_size - 1:
					self.link_array[link_idx]['node_1'] = np.int16(node_1)
					self.link_array[link_idx]['node_2'] = np.int16(node_2)
					link_idx = link_idx + 1
				else:
					print('ERROR! ran out of room in the link array for intra-plane links')
					return None

		self.number_of_gnd_links = link_idx - self.number_of_isl_links
		self.total_links = link_idx
