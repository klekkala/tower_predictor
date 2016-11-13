import shapely.geometry as geom
import numpy as np
import matplotlib as mplPath

def point_distance(x, y, towerx, towery):

	return np.sqrt((x2[0] - x1[0])**2 + (y2[1] - y1[1])**2)


def point_polygon(x, polygon, value):

	for unit in polygon:
		bbPath = mplPath.Path(unit)
		if bbPath.contains_point(x) == 1:
			break

	return value[i]

#def point_line(x, line_coords):
	#coords = np.loadtxt('points.txt')

	#line = geom.LineString(line_coords)
	#point = geom.Point(x)

	# Note that "line.distance(point)" would be identical
	#return point.distance(line)

def point_interpolate(x, y, elevx, elevy, elevation):
	xel = np.array(elevx)
	yel = np.array(elevy)
	xco = np.array(x)
	yco = np.array(y)

	xxel, yyel = np.meshgrid(xel, yel)
	z = np.array(elevation)
	f = interpolate.interp2d(xel, yel, z, kind='cubic')
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
	y_pop = np.array(pop)
	y_elev = numpy.full(len(elev), max(elev))
	y_cost = numpy.full(len(land), min(land))


	y_value = numpy.concatenate(y_tower, y_pop, y_elev, y_land)
	return y_value