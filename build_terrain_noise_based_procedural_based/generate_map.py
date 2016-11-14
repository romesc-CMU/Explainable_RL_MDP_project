from __future__ import division
import math, random, copy, csv, datetime, os
import numpy as np
# import matplotlib.pyplot as plt
from sets import Set

from tensor_field import GridTensorField, RadialTensorField, BoundaryTensorField
from utils import distance_pt_line, get_min_index, in_hull
from generate_map_utils import weighted_evec_after_rk4, add_area
from drawer import Drawer

import logging
log_base_path = './'
log_dir = os.path.join(log_base_path, 'logging_python.log')
# basic logger level
logger = logging.getLogger('navigation_explanation')
# fh's and ch's logging level must be higher than this basic logger level
# Because only if this basic logger records the msg, fh and ch can get it from this basic logger
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file handler level
fh = logging.FileHandler(log_dir, mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
# console handler level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
# logger.critical('This is a critical message.')
# logger.error('This is an error message.')
# logger.warning('This is a warning message.')
# logger.info('This is an informative message.')
# logger.debug('This is a low-level debug message.')


# TODO: empty the roads inside water and popu center
# TODO: make boundary tensor around water and popu center
# TODO: connect each road so that all of them are reachable
inf = 1000.
num_benchmarks = 40
# num_benchmarks = 2

road = 1
obstacle = 2

tree = 11
person = 12
obstacle_conditions = [tree, person]

gravel = 21
grass = 22
road_conditions = [road, gravel, grass]


if __name__ == "__main__":
    # drawer = Drawer()
    # colors = drawer.get_colors()
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')

    for shen2 in xrange(1):
        for shen in xrange(num_benchmarks):
            # try:
            if True:
                random.seed()
                num_row = (shen2+1)*10
                num_col = num_row*2
                print num_row,' ',num_col
         
                decay_const = random.sample([0.01,0.1,1.0],1)[0]
                # if shen < num_benchmarks*0.4:
                #     max_coverage = 0.2
                # elif shen < num_benchmarks*(0.4+0.28):
                #     max_coverage = 0.25
                # elif shen < num_benchmarks*(0.4+0.28+0.2):
                #     max_coverage = 0.3
                # else:
                #     max_coverage = 0.35
                max_coverage = 0.35
                
                # num_row = 10
                # num_col = 20
                # max_coverage = 0.2
                # decay_const = 1.

                max_num_grid_tf = 5
                max_num_radial_tf = 5
                max_num_boundary_tf = 5
                max_len_polyline = 5
                max_num_obstacle = 60


                # trace roads
                # min_pts_in_hyperstreamline = int(0.03*num_row*num_col)
                min_pts_in_hyperstreamline = int(0.01*num_row*num_col)
                min_dist_between_roads = 1.
                min_dist_between_road_obstacle = 1.

                # we will sample points inside a quadrilateral. 
                # We need to define the quadrilateral max size
                max_obstacle_length = 0.4*num_col 
                max_obstacle_width = 0.2*num_row
                max_num_obstacle_samples = max_obstacle_length*max_obstacle_width
                # XXX: expected obstacles = 20*0.2*20*0.1*20 / (20*40) = 160/800 = 0.20
                max_num_obstacles = 40
                # max_num_obstacles = 0

                # 1. Build terrains
                terrains = np.zeros((num_row, num_col), dtype=np.int)

                # (1) add obstacle
                # obstacles_hull_points, obstacles_hull_simplices = add_area\
                #         (max_obstacle_length, max_obstacle_width,\
                #         max_num_obstacle_samples, max_num_obstacles, \
                #         terrains, num_col, num_row, obstacle, fig, ax)
                obstacles_hull_points, obstacles_hull_simplices = add_area\
                        (max_obstacle_length, max_obstacle_width,\
                        max_num_obstacle_samples, max_num_obstacles, \
                        terrains, num_col, num_row, obstacle)

                # 2. Random Tensor Field Generation
                # each ts must be present once
                tss = []
                dominant_tf = np.zeros((num_row, num_col), dtype=np.int)
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

                # add boundary for obstacles
                for i in xrange(len(obstacles_hull_simplices)):
                    boundary_tf = BoundaryTensorField(obstacles_hull_points[i][:,0], \
                            obstacles_hull_points[i][:,1])
                    tss.append(boundary_tf) 
                    boundary_tfs.append(boundary_tf)

                    # XXX: now we want all the regions are gravel, instead of persons and trees
                    # obs_cond = random.sample(obstacle_conditions,1)
                    obs_cond = [gravel]

                    for m in xrange(num_row):
                        for n in xrange(num_col):
                            dists = []
                            for simplices in obstacles_hull_simplices[i]:
                                dists.append(distance_pt_line(simplices[0][0], simplices[0][1],\
                                        simplices[1][0], simplices[1][1], n, m))
                            min_dist = min(dists)
                            min_index = dists.index(min_dist)
                            if min_dist <= min_dist_between_road_obstacle\
                                    or in_hull([[n,m]],obstacles_hull_points[i]):
                                dominant_tf[m][n] = len(tss)-1
                            if in_hull([[n,m]],obstacles_hull_points[i]):
                                terrains[m][n] = obs_cond[0]

                # for i in xrange(num_boundary_tfs):
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
                        if dominant_tf[i][j] == 0:
                            dominant_tf[i][j] = int(random.randint(0,num_tss-1))
                logger.info('dominant_tf='+str(dominant_tf))
                logger.info('terrains='+str(terrains))

                # 3. Street Graph Generation
                # (1) find the first seed
                seed_weights = np.zeros((num_row, num_col))
                for i in xrange(num_row):
                    for j in xrange(num_col):
                        if terrains[i][j] != obstacle:
                            dbs = []
                            for simplices in obstacles_hull_simplices:
                                for simplice in simplices:
                                    dbs.append(distance_pt_line(simplice[0][0], simplice[0][1],\
                                            simplice[1][0], simplice[1][1], j, i))
                            if len(dbs)==0:
                                db = 0.
                            else:
                                db = min(dbs)
                            seed_weights[i][j] = math.pow(math.e, -db)
                        else:
                            seed_weights[i][j] = inf

                # print seed_weights
                seed_val, seed_row, seed_col = get_min_index(seed_weights)
                seed_weights[seed_row][seed_col] = inf

                # (2) RK4 to trace Hyperstreamline
                # x - j, y - i
                weighted_tf_xs = np.zeros((num_row, num_col))
                weighted_tf_ys = np.zeros((num_row, num_col))

                # hyperstreamlines_pts, hyperstreamlines are all in [row,col]
                hyperstreamlines_pts = Set([])
                hyperstreamlines = []
                # keep track of all the seeds
                was_seed_before = np.zeros((num_row, num_col))

                # we compare road/non_obs and cur_coverage to stop

                num_non_obs = 0
                for i in xrange(num_row):
                    for j in xrange(num_col):
                        if terrains[i][j] != tree and terrains[i][j] != person:
                            num_non_obs += 1
                # print num_non_obs
                # print num_row*num_col
                         

                major = True
                while True:
                    # print '>>>>>>>>>>>>>>>>>>>>>>'
                    # print 'new seed_row=', seed_row
                    # print 'new seed_col=', seed_col
                    target_i = seed_row
                    target_j = seed_col
                    was_seed_before[seed_row][seed_col] = True

                    target_i_origin = copy.deepcopy(seed_row)
                    target_j_origin = copy.deepcopy(seed_col)
                    prev_direction = np.array([0.,0.])
                    points = [[target_i, target_j]]
                   
                    while True:
                        # print '-----------------------------'
                        # print 'cur_target_i=',target_i
                        # print 'cur_target_j=',target_j

                        # new_direction, prev_direction are in [x,y]
                        new_direction = weighted_evec_after_rk4(major, prev_direction,\
                                tss, dominant_tf, target_i, target_j, num_row, num_col, decay_const)

                        if np.linalg.norm(new_direction) > math.pow(10.,-3):
                            new_direction = new_direction/np.linalg.norm(new_direction)
                        # print 'new_direction=',new_dir
                        # print 'target_i',target_i
                        # print 'target_j',target_j
                        weighted_tf_xs[target_i][target_j] = new_direction[0]
                        weighted_tf_ys[target_i][target_j] = new_direction[1]

                        # next_targets = np.array([np.array([-1.,-1.]), np.array([-1.,0.]),\
                                # np.array([-1.,1.]), np.array([0.,-1.]), np.array([0.,1.]),\
                                # np.array([1.,-1.]), np.array([1.,0.]), np.array([1.,1.])])
                        next_targets = np.array([np.array([-1.,0.]), np.array([1.,0.]),\
                                np.array([0.,-1.]), np.array([0.,1.])])

                        dists = [np.linalg.norm(xxx-new_direction) for xxx in next_targets]
                        # print 'dists=',dists
                        closest_neighbor_index = dists.index(min(dists))
                        closest_neighbor = next_targets[closest_neighbor_index]

                        # closest_neighbor is in [x,y]
                        target_i = closest_neighbor[1]+target_i
                        target_j = closest_neighbor[0]+target_j
                        # print 'new target_i=',target_i
                        # print 'new target_j=',target_j

                        # we need to make sure that around new target, there is no road
                        # because we want to keep road far from each other
                        road_surrounded = False
                        non_road_pts = []
                        for i in xrange(num_row):
                            for j in xrange(num_col):
                                if terrains[i][j] == road and [i,j] not in points:
                                    non_road_pts.append([i,j])
                        for pt in non_road_pts:
                            if np.linalg.norm([pt[0]-target_i,pt[1]-target_j]) <= min_dist_between_roads:
                                road_surrounded = True
                                break
                        if road_surrounded == True:
                            # print '1'
                            break
                        if target_i < 0 or target_i >= num_row or target_j < 0 or target_j >= num_col:
                            # print '2'
                            break
                        elif target_i == target_i_origin and target_j == target_j_origin:
                            # print '3'
                            break
                        # points are in [row,col]
                        points.append([target_i,target_j])
                        prev_direction = new_direction

                    major = not major

                    # print 'points', points

                    # if there is a road in this hyperstreamline
                    if len(points) >= min_pts_in_hyperstreamline:
                        hyperstreamline_is = [e[0] for e in points]
                        hyperstreamline_js = [e[1] for e in points]
                        # ax.plot(hyperstreamline_js, hyperstreamline_is,\
                                # color=colors['red'], linestyle='-', linewidth=2)
                        for h_pts in points:
                            # print h_pts
                            terrains[h_pts[0]][h_pts[1]] = road
                            hyperstreamlines_pts.add(tuple(h_pts))
                        hyperstreamlines.append(points)
                    
                    new_seeds = []
                    for p in hyperstreamlines_pts:
                        if was_seed_before[p[0]][p[1]] == False: 
                            new_seeds.append([p[0],p[1]])

                    # new_seeds = []
                    # for row_offset in [-1,0,1]:
                        # for col_offset in [-1,0,1]:
                            # row = seed_row+row_offset
                            # col = seed_col+col_offset
                            # if row >= 0 and row < num_row and col >= 0 and col < num_col\
                                    # and was_seed_before[row][col] == False:
                                    # new_seeds.append([row, col])
                    if len(new_seeds) <= 0:
                        # print 'new_seeds = []'
                        seed_val_tmp, seed_row_tmp, seed_col_tmp = get_min_index(seed_weights)
                        seed_weights[seed_row_tmp][seed_col_tmp] = inf
                        new_seed = [[seed_row_tmp, seed_col_tmp]]
                    else:
                        new_seed = random.sample(new_seeds,1)
                    if new_seed[0][0] == seed_row and new_seed[0][1] == seed_col:
                        # print 'duplicate seed'
                        seed_val, seed_row, seed_col = get_min_index(seed_weights)
                        seed_weights[seed_row][seed_col] = inf
                    else:
                        seed_row = new_seed[0][0]
                        seed_col = new_seed[0][1]

                    # Check if we have enough nodes visited by looking at the set hyperstreamlines_pts 
                    cur_coverage = len(hyperstreamlines_pts)/(num_non_obs)
                    print 'cur_coverage',cur_coverage 
                    if cur_coverage >= max_coverage:
                        break
                # XXX: now we want the road be only road, not gravel or grass road
                for pts in hyperstreamlines:
                    # road_cond = random.sample(road_conditions,1)
                    # for pt in pts: 
                        # terrains[pt[0]][pt[1]] = road_cond[0]
                    terrains[pt[0]][pt[1]] = road

                # XXX: all empty space is grass
                for i in xrange(num_row):
                    for j in xrange(num_col):
                        if terrains[i][j] == 0:
                            terrains[i][j] = grass

                # XXX: we randomly put person and trees on grass
                for _ in xrange(max_num_obstacle):
                    i = random.randint(0,num_row-1)
                    j = random.randint(0,num_col-1)
                    o = random.sample([tree,person],1)[0]
                    if terrains[i][j] == grass:
                       terrains[i][j] = o

                with open('./output/map_'+str(shen2)+'_'+str(shen)+'.csv', 'wb') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',')
                    # make output align with the plot
                    for t in reversed(terrains):
                        spamwriter.writerow(t)
                    spamwriter.writerow([decay_const])
                    spamwriter.writerow([max_num_obstacles])
                    spamwriter.writerow([max_coverage])
                    spamwriter.writerow([num_row])
                    spamwriter.writerow([num_col])
 
                
                # 4. Draw
                # for i in xrange(num_row):
                    # for j in xrange(num_col):
                #         if weighted_tf_xs[i][j]<math.pow(10.,-3)\
                                # and weighted_tf_ys[i][j]<math.pow(10.,-3)\
                                # and weighted_minor_xs[i][j]<math.pow(10.,-3)\
                #                 and weighted_minor_ys[i][j]<math.pow(10.,-3):
                            # ax.plot([j],[i], color='gray',marker='o')
                        # else:
                            # ax.plot([j, j+weighted_tf_xs[i][j]],\
                                    # [i,i+weighted_tf_ys[i][j]],\
                                    # color=colors['green'], linestyle='-', linewidth=2)
                            # ax.plot([j, j+weighted_minor_xs[i][j]],\
                                    # [i,i+weighted_minor_ys[i][j]],\
                                    # color=colors['yellow'], linestyle='-', linewidth=2)

                # for pts in hyperstreamlines_pts:
                #     ax.plot(pts[1],pts[0], color='black',marker='o')
                # for i in xrange(num_row):
                #     for j in xrange(num_col):
                #         if terrains[i][j] == tree:
                #             ax.plot(j,i, color='pink',marker='o')
                #         elif terrains[i][j] == person:
                #             ax.plot(j,i, color='orange',marker='o')

                # ax.set_xlim([0, num_col])
                # ax.set_ylim([0, num_row])
                # drawer.show_plt(fig, './output/map_'+str(shen2)+'_'+str(shen)+'.png', show=False)
                logger.info('image saved at ./output/map_'+str(shen2)+'_'+str(shen)+'.png')

                break
            # except:
            #     logger.info('error.')


