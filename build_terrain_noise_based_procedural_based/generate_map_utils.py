import random,math
import numpy as np
from noise import pnoise2

from scipy.spatial import ConvexHull

from utils import in_hull

from drawer import Drawer
drawer = Drawer()
colors = drawer.get_colors()


road = 1
obstacle = 2



# def add_area(max_length, max_width, max_num_samples, max_num_areas,\
#         terrains, num_col, num_row, terrain_type, fig, ax):
def add_area(max_length, max_width, max_num_samples, max_num_areas,\
        terrains, num_col, num_row, terrain_type):
    num_areas = random.randint(0,max_num_areas)
    # this is the apex of the triangle where we will sample the area
    apexs = []
    # the contour of convex hull represented in multiple polyline slices
    hull_contour_simplices = []
    # the contour of convex hull represented in multiple points (used for function in_hull)
    hull_contour_points = []
    for i in xrange(num_areas):
        # we keep the area in a quadrilateral, otherwise it might get too big
        # (1) sample the apex of the quadrilateral
        apex = np.array([random.random()*num_col, random.random()*num_row])
        tmp_x = apex[0]-max_length+random.random()*2.*max_length
        if tmp_x>=num_col or tmp_x<=0:
            if abs(tmp_x-num_col) < abs(tmp_x-0.):
                tmp_x = num_col
            else:
                tmp_x = 0.
        tmp_y = apex[1]-max_width+random.random()*2.*max_width
        if tmp_y>=num_row or tmp_y<=0:
            if abs(tmp_y-num_row) < abs(tmp_y-0.):
                tmp_y = num_row
            else:
                tmp_y = 0.
        apex2 = np.array([tmp_x,tmp_y])

        tmp_x = apex[0]-max_length+random.random()*2.*max_length
        if tmp_x>=num_col or tmp_x<=0:
            if abs(tmp_x-num_col) < abs(tmp_x-0.):
                tmp_x = num_col
            else:
                tmp_x = 0.
        tmp_y = apex[1]-max_width+random.random()*2.*max_width
        if tmp_y>=num_row or tmp_y<=0:
            if abs(tmp_y-num_row) < abs(tmp_y-0.):
                tmp_y = num_row
            else:
                tmp_y = 0.
        apex3 = np.array([tmp_x,tmp_y])
        # ax.plot(apex[0],apex[1], color=colors['green'],marker='o')
        # ax.plot(apex2[0],apex2[1], color=colors['green'],marker='o')
        # ax.plot(apex3[0],apex3[1], color=colors['green'],marker='o')

        # (2) sample the points of the area inside this quadrilateral
        # http://mathworld.wolfram.com/TrianglePointPicking.html
        v1 = apex2-apex
        v2 = apex3-apex
        num_samples = random.randint(3,max_num_samples)
        contour_pts = []
        for x in xrange(num_samples):
            sample = random.random()*v1+random.random()*v2+apex
            # ax.plot(sample[0],sample[1], color='white',marker='o')
            contour_pts.append(np.array(sample))
        contour_pts = np.array(contour_pts)
        hull = ConvexHull(contour_pts)
        hull_contour_points.append(np.array([contour_pts[s] for s in hull.vertices]))
        hull_contour_simplices.append(np.array\
                ([[contour_pts[s[0]], contour_pts[s[1]]] for s in hull.simplices]))

        # (4) update terrains 
        for m in xrange(num_row):
            for n in xrange(num_col):
                for w in hull_contour_points:
                    if in_hull([[n,m]],w):
                        terrains[m][n] = terrain_type 
        # print terrains

        # (3) draw the area
        color = ''
        if terrain_type == obstacle:
            color = 'blue'
        # ax.plot(hull_contour_points[i][:,0], hull_contour_points[i][:,1], \
        #         color=colors[color],marker='o')
        # for s in hull_contour_simplices[i]:
        #     ax.plot(s[:,0], s[:,1], color=colors[color], linestyle='-', linewidth=2)
    return hull_contour_points,hull_contour_simplices


# major_evec and minor_evec are in [x,y], target_i and target_j are in [row,col]
def weighted_sum_tensor_field(tss, dominant_types, \
    target_i, target_j, num_row, num_col, decay_const):
    dominant_type_org = dominant_types[target_i][target_j]
    major_evec = np.array([0.,0.])
    minor_evec = np.array([0.,0.])
    if dominant_type_org != -1: 
        for m in xrange(num_row):
            for n in xrange(num_col):
                dominant_type = dominant_types[m][n]
                [major,minor] = tss[dominant_type].tensor_field(target_j,target_i)
                # XXX major or minor eigenvector? which to use? 
                # a = math.pow(math.e,-1.*decay_const*(abs(m-target_i)*abs(m-target_i)+abs(n-target_j)*target_j(n-target_j)))
                # if a>math.pow(10.,-3):
                    # print a

                major_evec[0] += major[0]*math.pow(math.e,\
                        -1.*decay_const*(abs(m-target_i)*abs(m-target_i)+abs(n-target_j)*abs(n-target_j)))
                major_evec[1] += major[1]*math.pow(math.e,\
                        -1.*decay_const*(abs(m-target_i)*abs(m-target_i)+abs(n-target_j)*abs(n-target_j)))
                minor_evec[0] += minor[0]*math.pow(math.e,\
                        -1.*decay_const*(abs(m-target_i)*abs(m-target_i)+abs(n-target_j)*abs(n-target_j)))
                minor_evec[1] += minor[1]*math.pow(math.e,\
                        -1.*decay_const*(abs(m-target_i)*abs(m-target_i)+abs(n-target_j)*abs(n-target_j)))

        # (2) normalize tensor vector
        major_evec = [major_evec[0]/np.linalg.norm(major_evec),\
                major_evec[1]/np.linalg.norm(major_evec)]
        minor_evec = [minor_evec[0]/np.linalg.norm(minor_evec),\
                minor_evec[1]/np.linalg.norm(minor_evec)]

        # (3) add rotational field (use Perlin noise to generate the angle)
        # https://github.com/caseman/noise
        # http://nullege.com/codes/search/noise.pnoise2
        # Perlin noise is not actually "random" 
        # there are lots of easily discernible patterns
        # http://stackoverflow.com/questions/12473434/whats-the-randomness-quality-of-the-perlin-simplex-noise-algorithms

        R1_major = pnoise2(major_evec[0], major_evec[1])*math.pi-math.pi/2.
        R1_minor = -1.*R1_major
        major_evec = np.dot(np.array([major_evec[0],major_evec[1]]),\
                np.array([[math.cos(R1_major),-math.sin(R1_major)],\
                [math.sin(R1_major),math.cos(R1_major)]]))
        minor_evec = np.dot(np.array([minor_evec[0],minor_evec[1]]),\
                np.array([[math.cos(R1_minor),-math.sin(R1_minor)],\
                [math.sin(R1_minor),math.cos(R1_minor)]]))

        R2 = pnoise2(major_evec[0], major_evec[1])*math.pi-math.pi/2.
        major_evec = np.dot(np.array([major_evec[0],major_evec[1]]),\
                np.array([[math.cos(R2),-math.sin(R2)],\
                [math.sin(R2),math.cos(R2)]]))
        R3 = pnoise2(major_evec[0], major_evec[1])*math.pi-math.pi/2.
        minor_evec = np.dot(np.array([minor_evec[0],minor_evec[1]]),\
                np.array([[math.cos(R3),-math.sin(R3)],\
                [math.sin(R3),math.cos(R3)]]))

        # TODO: (4) Laplacian smoothing
    return major_evec, minor_evec


# When tracing major eigenvectors
# we need to remove the sign ambiguity in eigenvector directions
# prev_dir, new_evec, major_evec and minor_evec are in [x,y], target_i and target_j are in [row,col]
def weighted_evec(major, prev_dir, tss, dominant_types, \
    target_i, target_j, num_row, num_col, decay_const):

    major_evec, minor_evec = weighted_sum_tensor_field(tss,\
        dominant_types, target_i, target_j, num_row, num_col, decay_const)

    if major == True:
        evec = major_evec
    else:
        evec = minor_evec

    new_evec = np.array([0.,0.])
    # http://martindevans.me/game-development/2015/12/11/Procedural-Generation-For-Dummies-Roads/
    # If previous is zero that's a degenerate case, just bail out
    # Dot product >= zero indicates angle < 90
    if (prev_dir[0]<=math.pow(10.,-3) and prev_dir[1]<=math.pow(10.,-3))\
            or (np.dot(prev_dir, evec) > 0):
        new_evec = evec
    else:
        new_evec = -1.*evec
    return new_evec


# k's are in [x,y], target_i and target_j are in [row,col]
def weighted_evec_after_rk4(major, prev_dir, tss, dominant_types, \
    target_i, target_j, num_row, num_col, decay_const):
    # http://gafferongames.com/game-physics/integration-basics/
    # http://martindevans.me/game-development/2015/12/11/Procedural-Generation-For-Dummies-Roads/
    k1 = weighted_evec(major, prev_dir, tss, dominant_types,\
            target_i, target_j, num_row, num_col, decay_const)
    k2 = weighted_evec(major, prev_dir, tss, dominant_types,\
            target_i+k1[0]/2.,\
            target_j+k1[1]/2.,
            num_row, num_col, decay_const)
    k3 = weighted_evec(major, prev_dir, tss, dominant_types,\
            target_i+k2[0]/2.,\
            target_j+k2[1]/2.,\
            num_row, num_col, decay_const)
    k4 = weighted_evec(major, prev_dir, tss, dominant_types,\
            target_i+k3[0],\
            target_j+k3[1],\
            num_row, num_col, decay_const)
    return k1/6. + k2/3. + k3/3. + k4/6.

