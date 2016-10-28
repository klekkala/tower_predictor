import shapely.geometry as geom
import numpy as np
import matplotlib as mplPath

def point_distance(x1, x2):

	distance = sqrt((x2[0] - x1[0])**2 + (y2[1] - y1[1])**2)
	return distance


def point_polygon(x, polygon):

	for i in range(1,len(polygon)):
		bbPath = mplPath.Path(polygon)

	bbPath.contains_point(x)
	return value

def point_line(x, line_coords):
	coords = np.loadtxt('points.txt')

	line = geom.LineString(line_coords)
	point = geom.Point(x)

	# Note that "line.distance(point)" would be identical
	return point.distance(line)



def total_point_distance(x, points):
	for i in range(1, len(points)):
		total_sum = total_sum+point_distance(x, points[i])

	return total_sum


def val_point_polygon(x, line_list):

	min_val = 100
	for i in range(1, len(line_list)):
		min_val = min(min_val, point_line(x, line_list[i]))

	return min_val.value()


def min_point_line(x, polygon_list):
	
	for i in range(1, len(polygon_list)):
		if point_polygon(x, polygon_list[i]) == 1:
			return polygon_list[i].value()