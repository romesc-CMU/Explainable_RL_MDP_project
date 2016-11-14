import math
import numpy as np
from utils import distance_pt_line, matrix_to_eigen

class GridTensorField:
    def __init__(self, ux, uy):
        self.l = math.sqrt(ux*ux + uy*uy)
        self.theta = np.arctan2(uy,ux)
        # self.draw_length = 1.
        print 'Grid_tf l=', self.l
        print 'Grid_tf theta=', self.theta

    def tensor_field(self, x, y):
        # w,v = np.linalg.eig([[self.l*math.cos(2.*self.theta), 
            # self.l*math.sin(2.*self.theta)], 
            # [self.l*math.sin(2.*self.theta), 
                # -1.*self.l*math.cos(2.*self.theta)]])
        [evals, major, minor] = \
                matrix_to_eigen([[self.l*math.cos(2.*self.theta),\
                self.l*math.sin(2.*self.theta)],\
                [self.l*math.sin(2.*self.theta),\
                -1.*self.l*math.cos(2.*self.theta)]])
        # ret_vecs = []
        # ret_vecs.append([x+self.draw_length*major[0],y+self.draw_length*major[1]])
        # ret_vecs.append([x+self.draw_length*minor[0],y+self.draw_length*minor[1]])
        return [major,minor] 
"""
grid_tf = GridTensorField(1.4, 1.4)
for i in xrange(10):
    for j in xrange(10):
        point = [i,j]
        [major, minor] = grid_tf.tensor_field(point[0],point[1])
        plt.plot([point[0], point[0]+major[0]], [point[1], point[1]+major[1]] , 'b-', lw=2)
        plt.plot([point[0], point[0]+minor[0]], [point[1], point[1]+minor[1]] , 'k-', lw=2)
plt.show()
exit(0)
"""

               
class RadialTensorField:
    def __init__(self, x0, y0):
        self.x0 = x0
        self.y0 = y0
        print 'Radial_tf x0=', self.x0
        print 'Radial_tf y0=', self.y0

    def tensor_field(self, xp, yp):
        # import IPython;IPython.embed()
        x = float(xp-self.x0)
        y = float(yp-self.y0)
        # w,v = np.linalg.eig([[y*y-x*x, -2.*x*y],[-2.*x*y, -1.*(y*y-x*x)]])
        [evals, major, minor] = \
                matrix_to_eigen([[y*y-x*x, -2.*x*y],[-2.*x*y, -1.*(y*y-x*x)]])
        # ret_vecs = []
        # ret_vecs.append([x+self.draw_length*major[0],y+self.draw_length*major[1]])
        # ret_vecs.append([x+self.draw_length*minor[0],y+self.draw_length*minor[1]])
        return [major,minor] 
 
"""
radial_center = [50,50]
radial_tf = RadialTensorField(radial_center[0], radial_center[1])

# point = [10,50]
# [major,minor] = radial_tf.tensor_field(point[0],point[1])
# plt.plot([point[0], point[0]+major[0]], [point[1], point[1]+major[1]] , 'b-', lw=2)
# plt.plot([point[0], point[0]+minor[0]], [point[1], point[1]+minor[1]] , 'k-', lw=2)
# plt.plot([radial_center[0]], [radial_center[1]], 'ro')
# plt.axis('equal')
# plt.axis([0, 100, 0, 100])
# plt.show()
# exit(0)


plt.plot([radial_center[0]], [radial_center[1]], 'ro')
for i in xrange(100):
    for j in xrange(100):
        if i != radial_center[0] and j != radial_center[1]:
            point = [i,j]
            [major, minor] = radial_tf.tensor_field(point[0],point[1])
            plt.plot([point[0], point[0]+major[0]], [point[1], point[1]+major[1]] , 'b-', lw=2)
            # plt.plot([point[0], point[0]+minor[0]], [point[1], point[1]+minor[1]] , 'k-', lw=2)
            # plt.plot([point[0]], [point[1]], 'go')
plt.axis('equal')
plt.axis([0, 100, 0, 100])
plt.show()
exit(0)
"""


class BoundaryTensorField:
    def __init__(self, xs, ys):
        # input: a polyline of the brush (a list of x and y)
        # around that polyline, the directions are the same with the polyline
        # TODO: input should be a brushed area which will also use Laplacian equations
        self.xs = xs
        self.ys = ys
        print 'Boundary_tf xs=', self.xs
        print 'Boundary_tf ys=', self.ys

    def tensor_field(self, xp, yp):
        # for each sample point s, we will find the point c on the polyline
        # which is closest to s. Then the direction of s is equal to the polyline
        # at c
        # import IPython;IPython.embed()
        dists = [distance_pt_line(self.xs[i],self.ys[i], self.xs[i+1],self.ys[i+1], xp,yp) \
                for i in range(len(self.xs)-1)]
        ret_vecs = []
        if len(dists) == 0:
            print "BoundaryTensorField has no dists"
            exit(2)

        min_index = dists.index(min(dists))
        major = [self.xs[min_index+1]-self.xs[min_index], self.ys[min_index+1]-self.ys[min_index]]
        minor = [-1*self.ys[min_index+1]-self.ys[min_index], self.xs[min_index+1]-self.xs[min_index]]
        return [major,minor] 

"""
xs = [1,1,2,4]
ys = [1,2,4,4]
boundary_tf = BoundaryTensorField(xs,ys)
point = [2,1]
[major,minor] = boundary_tf.tensor_field(point[0],point[1])
plt.plot([point[0], point[0]+major[0]], [point[1], point[1]+major[1]] , 'b-', lw=2)
plt.plot([point[0], point[0]+minor[0]], [point[1], point[1]+minor[1]] , 'k-', lw=2)
plt.axis('equal')
plt.axis([0, 5, 0, 5])
plt.show()

for i in xrange(5):
    for j in xrange(5):
        point = [i,j]
        [major,minor] = boundary_tf.tensor_field(point[0],point[1])
        plt.plot([point[0], point[0]+major[0]], [point[1], point[1]+major[1]] , 'b-', lw=2)
        plt.plot([point[0], point[0]+minor[0]], [point[1], point[1]+minor[1]] , 'k-', lw=2)
        plt.plot([point[0]], [point[1]], 'go')
plt.show()
"""

