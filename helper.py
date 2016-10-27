import shapely.geometry as geom
import numpy as np


def point_distance(x1, x2):

	distance = sqrt((x2[0] - x1[0])**2 + (y2[1] - y1[1])**2)
	return distance


def point_polygon(x, polygon):

	for i in range(1,len(polygon)):
		bbPath = mplPath.Path(np.array([[poly[0], poly[1]],
                     [poly[1], poly[2]],
                     [poly[2], poly[3]],
                     [poly[3], poly[0]]]))

	bbPath.contains_point(x)
	return value

def point_line(x, line_coords):
	coords = np.loadtxt('points.txt')

	line = geom.LineString(line_coords)
	point = geom.Point(x)

	# Note that "line.distance(point)" would be identical
	return point.distance(line)
