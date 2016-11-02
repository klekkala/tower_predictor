import shapely.geometry as geom
import numpy as np
import matplotlib as mplPath

def point_distance(x, y, towerx, towery):

	return np.sqrt((x2[0] - x1[0])**2 + (y2[1] - y1[1])**2)


def point_polygon(x, polygon):

	for i in range(1,len(polygon)):
		bbPath = mplPath.Path(polygon)

	bbPath.contains_point(x)
	return value

#def point_line(x, line_coords):
	#coords = np.loadtxt('points.txt')

	#line = geom.LineString(line_coords)
	#point = geom.Point(x)

	# Note that "line.distance(point)" would be identical
	#return point.distance(line)

def point_interpolate(x, y, elevx, elevy)

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

	y_tower = np.array()
	y_pop = np.array()
	y_elev = np.array()
	y_land = np.array()

	for i in range(len(tower)):
		y_tower = np.append(tower[i].value)

	for i in range(len(tower)):
		y_pop = np.append(pop[i].value)

	for i in range(len(tower)):
		y_elev.append(elev[i].value)

	for i in range(len(land)):
		y_tower.append(land[i].value)

    y_value = numpy.concatenate(y_tower, y_pop, y_elev, y_land)

    return y_value