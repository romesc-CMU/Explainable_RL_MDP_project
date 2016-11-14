import numpy as np
import math
from scipy.spatial import Delaunay

# http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def distance_pt_line(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1
    something = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance
    dist = math.sqrt(dx*dx + dy*dy)
    return dist

# used for a traceless 2*2 matrix => 2 eigen vectors
def matrix_to_eigen(m):
    w,v = np.linalg.eig(m)
    assert(abs(m[0][0]-(-m[1][1]))<math.pow(10.,-3)\
            and abs(m[1][0]-(m[0][1]))<math.pow(10.,-3)) 
    assert(len(w)==2)
    assert(abs(w[0]+w[1])<math.pow(10.,-3))
    if abs(w[0]) <= math.pow(10.,-3):
    	print m
    assert(abs(w[0])>math.pow(10.,-3))
    if m[0][0] < math.pow(10.,-3):
        theta1 = math.pi/4.
        theta2 = math.pi*3./4.
        if abs(math.sin(2*theta1)-m[0][1]) < math.pow(10.,-3):
            theta = theta1
        else:
            theta = theta2

    theta = np.arctan2(m[0][1], m[0][0])/2.
    major = [abs(w[0])*math.cos(theta), abs(w[0])*math.sin(theta)]
    minor = [-abs(w[0])*math.sin(theta), abs(w[0])*math.cos(theta)]
    return [w,major,minor]



def in_hull(p, hull):
    # http://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


def get_min_index(matrix):
	matrix = np.array(matrix)
	min_num = 10000
	min_row = -1
	min_col = -1
	for i in xrange(matrix.shape[0]):
		for j in xrange(matrix.shape[1]):
			if matrix[i][j] < min_num:
				min_num = matrix[i][j]
				min_row = i
				min_col = j
	if min_num == 10000 or min_row == -1 or min_col == -1:
		print 'utils - min_index ERROR!'
		exit(0)
	return min_num, min_row, min_col

