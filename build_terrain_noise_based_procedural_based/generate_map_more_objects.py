from __future__ import division
import math, random, copy, csv
import numpy as np
import matplotlib.pyplot as plt
from sets import Set

from tensor_field import GridTensorField, RadialTensorField, BoundaryTensorField
from utils import distance_pt_line, get_min_index, in_hull
from map_gen_utils import weighted_major_evec_after_rk4, add_area
from drawer import Drawer

# TODO: empty the roads inside water and popu center
# TODO: make boundary tensor around water and popu center
# TODO: connect each road so that all of them are reachable
inf = 1000.

num_row =10 
num_col = 20


max_num_grid_tf = 5
max_num_radial_tf = 5
max_num_boundary_tf = 5
max_len_polyline = 5
decay_const = 0.1

max_coverage = 0.5
min_pts_in_hyperstreamline = int(0.03*num_row*num_col)
min_dist_between_roads = 0.
min_dist_between_road_water = 1.

# we will sample points inside a quadrilateral. We need to define the quadrilateral max size
max_water_length = num_col 
max_water_width = 0.4*num_row
max_num_water_samples = max_water_length*max_water_width
max_num_water_areas = 2

max_popu_center_length = 0.5*num_col 
max_popu_center_width = 0.5*num_row
max_num_popu_center_samples = max_popu_center_length*max_popu_center_width
max_num_popu_center_areas = 4

water = 1
road = 2
# house or forest
popu_center = 3

sand = 5
ice = 6
rock = 7
traffic = 8
unpaved = 9
grass = 10
road_conditions = [sand, ice, rock, traffic, unpaved, grass]

human = 99 
characters = [human]
max_num_characters = 5

if __name__ == "__main__":
    drawer = Drawer()
    colors = drawer.get_colors()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    random.seed()


    # 1. Build terrains
    terrains = np.zeros((num_row, num_col), dtype=np.int)
    # (1) add population center 
    hull_popu_center_contour_points, hull_popu_center_contour_simplices = add_area\
            (max_popu_center_length, max_popu_center_width,\
            max_num_popu_center_samples, max_num_popu_center_areas, \
            terrains, num_col, num_row, popu_center, fig, ax)

    # (2) add water
    hull_water_contour_points, hull_water_contour_simplices = add_area\
            (max_water_length, max_water_width,\
            max_num_water_samples, max_num_water_areas, \
            terrains, num_col, num_row, water, fig, ax)


    # 2. Random Tensor Field Generation
    # each ts must be present once
    # TODO: consider the water and popu_center
    tss = []
    dominant_types = np.zeros((num_row, num_col), dtype=np.int)
    grid_tfs = []
    radial_tfs = []
    boundary_tfs = []

    num_tss = 0
    num_grid_tfs = random.randint(1,max_num_grid_tf)
    num_radial_tfs = random.randint(1,max_num_radial_tf)
    num_boundary_tfs = random.randint(1,max_num_boundary_tf)

    for i in xrange(num_grid_tfs):
        # XXX: we use random instead of randomInt because we allow these to be not on the grid
        grid_tf = GridTensorField(random.random()*num_col, random.random()*num_row)
        tss.append(grid_tf) 
        grid_tfs.append(grid_tf)
    for i in xrange(num_radial_tfs):
        # XXX: we use random instead of randomInt because we allow these to be not on the grid
        radial_center = [random.random()*num_col, random.random()*num_row]
        radial_tf = RadialTensorField(radial_center[0], radial_center[1])
        tss.append(radial_tf) 
        radial_tfs.append(radial_tf)

    # add boundary for water and popu_center
    for i in xrange(len(hull_popu_center_contour_simplices)):
        boundary_tf = BoundaryTensorField(hull_popu_center_contour_points[i][:,0], \
                hull_popu_center_contour_points[i][:,1])
        tss.append(boundary_tf) 
        boundary_tfs.append(boundary_tf)

        for m in xrange(num_row):
            for n in xrange(num_col):
                dists = []
                for simplices in hull_popu_center_contour_simplices[i]:
                    dists.append(distance_pt_line(simplices[0][0], simplices[0][1],\
                            simplices[1][0], simplices[1][1], n, m))
                min_dist = min(dists)
                min_index = dists.index(min_dist)
                if min_dist <= min_dist_between_road_water \
                        or in_hull([[n,m]],hull_popu_center_contour_points[i]):
                    dominant_types[m][n] = len(tss)-1
 
    for i in xrange(len(hull_water_contour_simplices)):
        boundary_tf = BoundaryTensorField(hull_water_contour_points[i][:,0], \
                hull_water_contour_points[i][:,1])
        tss.append(boundary_tf) 
        boundary_tfs.append(boundary_tf)

        for m in xrange(num_row):
            for n in xrange(num_col):
                dists = []
                for simplices in hull_water_contour_simplices[i]:
                    dists.append(distance_pt_line(simplices[0][0], simplices[0][1],\
                            simplices[1][0], simplices[1][1], n, m))
                min_dist = min(dists)
                min_index = dists.index(min_dist)
                if min_dist < min_dist_between_road_water:
                    dominant_types[m][n] = len(tss)-1
 

#     for i in xrange(num_boundary_tfs):
        # xs = []
        # ys = []
        # # we need minimum 2 points in the polyline
        # tmp = random.randint(2,max_len_polyline)
        # for j in xrange(tmp):
            # # XXX: we use random instead of randomInt 
            # # because we allow these to be not on the grid
            # xs.append(random.random()*num_col)
            # ys.append(random.random()*num_row)
        # boundary_tf = BoundaryTensorField(xs,ys)
        # tss.append(boundary_tf) 
        # boundary_tfs.append(boundary_tf)

    num_tss = len(tss)

    
    for i in xrange(num_row):
        for j in xrange(num_col):
            # sometimes this position has road, sometimes doesn't
            # depend on road_coverage porobability
            # rand = random.random()
            # if rand-road_coverage<=math.pow(10.,-3):
            if dominant_types[i][j] == 0:
                dominant_types[i][j] = int(random.randint(0,num_tss-1))
            # else:
                # dominant_types[i][j] = -1
    print dominant_types


    # 3. Street Graph Generation
    # (1) find the first seed
    seed_weights = np.zeros((num_row, num_col))
    for i in xrange(num_row):
        for j in xrange(num_col):
            if terrains[i][j] != water:
                dbs = []
                for simplices in hull_water_contour_simplices:
                    for simplice in simplices:
                        dbs.append(distance_pt_line(simplice[0][0], simplice[0][1],\
                                simplice[1][0], simplice[1][1], j, i))

                db = min(dbs)

                dps = []
                for simplices in hull_popu_center_contour_simplices:
                    for simplice in simplices:
                        dps.append(distance_pt_line(simplice[0][0], simplice[0][1],\
                                simplice[1][0], simplice[1][1], j, i))
                dp = min(dps)
                seed_weights[i][j] = math.pow(math.e, -db)+math.pow(math.e, -dp)
            else:
                seed_weights[i][j] = inf

    # print seed_weights
    seed_num, seed_row, seed_col = get_min_index(seed_weights)
    seed_weights[seed_row][seed_col] = inf

    # (2) RK4 to trace Hyperstreamline
    # because 2D array uses [row][col], each point's tensor 
    # = (weighted_major_xs[i][j]-j, weighted_major_ys[i][j]-i)
    # but != (weighted_major_xs[i][j]-i, weighted_major_ys[i][j]-j)
    weighted_major_xs = np.zeros((num_row, num_col))
    weighted_major_ys = np.zeros((num_row, num_col))
    weighted_minor_xs = np.zeros((num_row, num_col))
    weighted_minor_ys = np.zeros((num_row, num_col))
    hyperstreamlines = Set([])
    hyperstreamlines_pts = []
    # keep track of all the seeds
    was_seed_before = np.zeros((num_row, num_col))

    while True:
        # print '>>>>>>>>>>>>>>>>>>>>>>'
        # print 'new seed_row=', seed_row
        # print 'new seed_col=', seed_col
        target_i = seed_row
        target_j = seed_col
        was_seed_before[seed_row][seed_col] = 1

        target_i_origin = copy.deepcopy(seed_row)
        target_j_origin = copy.deepcopy(seed_col)
        prev_dir = np.array([0.,0.])
        hyperstreamline_pts = [[target_i, target_j]]
       
        while True:
            # print '-----------------------------'
            # print 'cur_target_i=',target_i
            # print 'cur_target_j=',target_j
            new_dir = weighted_major_evec_after_rk4(prev_dir, tss, dominant_types,\
                    target_i, target_j, num_row, num_col, decay_const)
            if np.linalg.norm(new_dir) > math.pow(10.,-3):
                    new_dir = new_dir/np.linalg.norm(new_dir)
            # print 'new_dir=',new_dir
            # print 'target_i',target_i
            # print 'target_j',target_j
            target_i = int(target_i)
            target_j = int(target_j)
            weighted_major_xs[target_i][target_j] = new_dir[0]
            weighted_major_ys[target_i][target_j] = new_dir[1]

            next_targets = np.array([np.array([-1.,-1.]), np.array([-1.,0.]),\
                    np.array([-1.,1.]), np.array([0.,-1.]), np.array([0.,1.]),\
                    np.array([1.,-1.]), np.array([1.,0.]), np.array([1.,1.])])
            dists = [np.linalg.norm(new_target-new_dir) for new_target in next_targets]
            # print 'dists=',dists
            closest_neighbor_index = dists.index(min(dists))
            closest_neighbor = next_targets[closest_neighbor_index]

            target_i = closest_neighbor[0]+target_i
            target_j = closest_neighbor[1]+target_j
            # print 'new target_i=',target_i
            # print 'new target_j=',target_j

            # we need to make sure that around new target, there is no road
            # because we want to keep road far from each other
            road_surrounded = False
            for i in xrange(num_row):
                for j in xrange(num_col):
                    if np.linalg.norm([i-target_i,j-target_j]) <= min_dist_between_roads \
                            and terrains[i][j] == road:
                        road_surrounded = True
                        break
            if road_surrounded == True:
                break
 
            if target_i < 0 or target_i >= num_row or target_j < 0 or target_j >= num_col:
                break
            elif target_i == target_i_origin and target_j == target_j_origin:
                break
            # elif terrains[target_i][target_j] == water:
                # break
            hyperstreamline_pts.append([target_i,target_j])
            prev_dir = new_dir

        # print 'hyperstreamline_pts', hyperstreamline_pts

        # if there is a road in this hyperstreamline
        if len(hyperstreamline_pts) >= min_pts_in_hyperstreamline:
            # Draw it 
            hyperstreamline_xs = [e[0] for e in hyperstreamline_pts]
            hyperstreamline_ys = [e[1] for e in hyperstreamline_pts]
            # ax.plot(hyperstreamline_ys, hyperstreamline_xs,\
                    # color=colors['red'], linestyle='-', linewidth=2)
            for h_pts in hyperstreamline_pts:
                # print h_pts
                terrains[h_pts[0]][h_pts[1]] = road
                hyperstreamlines.add(tuple(h_pts))

            hyperstreamlines_pts.append(hyperstreamline_pts)
        
        new_seeds = []
        for row_offset in [-1,0,1]:
            for col_offset in [-1,0,1]:
                if not (row_offset==0 and col_offset==0):
                    row = seed_row+row_offset
                    col = seed_col+col_offset
                    if row < 0 or row >= num_row or col < 0 or col >= num_col:
                        break
                    if was_seed_before[row][col] == 0:
                        new_seeds.append([row, col])
        if len(new_seeds) <= 0:
            seed_num_tmp, seed_row_tmp, seed_col_tmp = get_min_index(seed_weights)
            seed_weights[seed_row_tmp][seed_col_tmp] = inf
            new_seed = [[seed_row_tmp, seed_col_tmp]]
        else:
            new_seed = random.sample(new_seeds,1)
        if new_seed[0][0] == seed_row and new_seed[0][1] == seed_col:
            seed_num, seed_row, seed_col = get_min_index(seed_weights)
            seed_weights[seed_row][seed_col] = inf
        else:
            seed_row = new_seed[0][0]
            seed_col = new_seed[0][1]

        # Check if we have enough nodes visited by looking at the set hyperstreamlines 
        cur_coverage = len(hyperstreamlines)/(num_col*num_row)
        print 'cur_coverage',cur_coverage 
        if cur_coverage >= max_coverage:
            break


    
    for hyperstreamline_pts in hyperstreamlines_pts:
        road_cond = random.sample(road_conditions,1)
        for pt in hyperstreamline_pts: 
            terrains[pt[0]][pt[1]] = road_cond[0]

    for i in xrange(random.randint(0,max_num_characters)):
        col = int(random.random()*num_col)
        row = int(random.random()*num_row)
        if terrains[row][col] != water and terrains[row][col] != popu_center:
            terrains[row][col] = random.sample(characters,1)[0]
    print terrains

    with open('./terrains.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        # make output align with the plot
        for t in reversed(terrains):
            spamwriter.writerow(t)

    # 4. Draw
    # for i in xrange(num_row):
        # for j in xrange(num_col):
    #         if weighted_major_xs[i][j]<math.pow(10.,-3)\
                    # and weighted_major_ys[i][j]<math.pow(10.,-3)\
                    # and weighted_minor_xs[i][j]<math.pow(10.,-3)\
    #                 and weighted_minor_ys[i][j]<math.pow(10.,-3):
                # ax.plot([j],[i], color='gray',marker='o')
            # else:
                # ax.plot([j, j+weighted_major_xs[i][j]],\
                        # [i,i+weighted_major_ys[i][j]],\
                        # color=colors['green'], linestyle='-', linewidth=2)
                # ax.plot([j, j+weighted_minor_xs[i][j]],\
                        # [i,i+weighted_minor_ys[i][j]],\
                        # color=colors['yellow'], linestyle='-', linewidth=2)


    for pts in hyperstreamlines:
        ax.plot(pts[1],pts[0], color='black',marker='o')
    for i in xrange(num_row):
        for j in xrange(num_col):
            if terrains[i][j] == water:
                ax.plot(j,i, color='blue',marker='o')
            elif terrains[i][j] == popu_center:
                ax.plot(j,i, color='orange',marker='o')


        

    ax.set_xlim([0, num_col])
    ax.set_ylim([0, num_row])
    drawer.show_plt(fig, './map.png', show=True)

