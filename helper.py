import shapely.geometry as geom
import numpy as np
import matplotlib.path as mplPath
from scipy import interpolate

def point_distance(x, y, towerx, towery):

	return np.sqrt((towerx - x)**2 + (towery - y)**2)


def point_polygon(x, y, polygon, value):

	point = [x, y]
	for i in range(0,len(polygon)):
		bbPath = mplPath.Path(polygon[i])
		if bbPath.contains_point(point) == 1:
			break

	return np.array([value[i]])

#def point_line(x, line_coords):
	#coords = np.loadtxt('points.txt')

	#line = geom.LineString(line_coords)
	#point = geom.Point(x)

	# Note that "line.distance(point)" would be identical
	#return point.distance(line)

def point_interpolate(x, y, elevx, elevy, elevation):

## don't forget to change the shape
	z = np.zeros((5,5), dtype = np.float_)
	for i in range(0, len(elevation)):
		z[i, i] = elevation[i]
	f = interpolate.interp2d(elevx, elevy, z, kind='cubic')
	elev_value = f(x,y)
	return elev_value

def val_point_polygon(x, line_list):

	min_val = 100
	for i in range(1, len(line_list)):
		min_val = min(min_val, point_line(x, line_list[i]))

	return min_val.value()


#def min_point_line(x, polygon_list):
	
	#for i in range(1, len(polygon_list)):
		#if point_polygon(x, polygon_list[i]) == 1:
			#return polygon_list[i].value()


def feature_properties(tower, pop, elev, land):

	y_tower = np.array(tower)
	y_pop = np.full(4,0)
	y_elev = np.array([max(elev)])
	y_cost = np.array([min(land)])


	y_value = np.concatenate((y_tower, y_pop, y_elev, y_cost))
	return y_value