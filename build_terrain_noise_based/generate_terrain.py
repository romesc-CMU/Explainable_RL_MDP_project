from __future__ import division
import math, random, copy, csv, datetime, os
import numpy as np
# import matplotlib.pyplot as plt
from sets import Set

from generate_noise import compute_noise,island_noise
from generate_road import generate_bezier_road, \
        generate_horizontal_road, generate_vertical_road

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


road = 1
grass = 2
rock = 3
tree = 4
person = 5

# opts_history_window_size = 3


def generate_terrain(num_row, num_col, \
        person_threshold, tree_threshold, rock_threshold,
        num_horizontal_road, num_vertical_road, background):

    random.seed()
    print num_row,'x',num_col
    logger.info(str(num_row)+' x '+str(num_col))

    # random walk based road generation
    # max_num_hori_roads = 1
    # max_num_vert_roads = 1
    # the bigger, the less randomness
    # opts_history_prob_param = 0.5
    # assert(opts_history_prob_param*math.e>1.)

    # add road to terrains
    terrains = np.zeros((num_row, num_col), dtype=np.int)

    # generate random polygon
    # http://ilyasterin.com/blog/2010/05/random-points-in-polygon-generation-algorithm.html
    # generate points in polygon
    # http://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
    
    # 1. randomly generate 'rock islands' on the map
    # http://www.redblobgames.com/maps/terrain-from-noise/
    # we first generate rocks so the road will overwrite the rocks if need
    noise_values = island_noise(num_row/2,num_col/2,\
            num_row,num_col,a=0.05,b=1.,c=1.5)
    for y in xrange(num_row):
        for x in xrange(num_col):
            if noise_values[y][x] >= rock_threshold:
                terrains[y][x] = rock

    # 2. all empty space is grass
    for y in xrange(num_row):
        for x in xrange(num_col):
            if terrains[y][x] == 0:
                if background != None:
                    terrains[y][x] = background

    # 3(1) bezier curve based road generation
    roads_rows = []
    roads_cols = []
    num_hori_roads = 0
    num_vert_roads = 0
    for _ in xrange(num_horizontal_road):
        # XXX We prefer horizontal road than vertical
        road_rows, road_cols = \
                generate_bezier_road(num_row, num_col, \
                num_pt_bezier=4, horizontal=True)
        roads_rows.append(road_rows)
        roads_cols.append(road_cols)
        num_hori_roads += 1
    for _ in xrange(num_vertical_road):
        road_rows, road_cols = \
                generate_bezier_road(num_row, num_col, \
                num_pt_bezier=4, horizontal=False)
        roads_rows.append(road_rows)
        roads_cols.append(road_cols)
        num_vert_roads += 1

    for i in xrange(len(roads_rows)):
        for j in xrange(len(roads_rows[i])):
            terrains[roads_rows[i][j]]\
                    [roads_cols[i][j]] = road


    # 3(2) random walk based road generation
    # num_hori_roads = random.randint(1,max_num_hori_roads)
    # hori_roads_rows = []
    # hori_roads_cols = []
    # for i in xrange(num_hori_roads): 
        # # hori_road_rows, hori_road_cols = \
                # # generate_horizontal_road(num_row, num_col, \
                # # opts_history_window_size, opts_history_prob_param)
        # hori_roads_rows.append(hori_road_rows)
        # hori_roads_cols.append(hori_road_cols)
    # num_vert_roads = random.randint(1,max_num_vert_roads)
    # vert_roads_rows = []
    # vert_roads_cols = []
    # for i in xrange(num_vert_roads): 
        # # vert_road_rows, vert_road_cols = \
                # # generate_vertical_road(num_row, num_col, \
                # # opts_history_window_size, opts_history_prob_param)
        # vert_roads_rows.append(vert_road_rows)
        # vert_roads_cols.append(vert_road_cols)
    # for road in xrange(len(hori_roads_rows)):
        # for pt in xrange(len(hori_roads_rows[road])):
            # terrains[hori_roads_rows[road][pt]]\
                    # [hori_roads_cols[road][pt]] = road
    # for road in xrange(len(vert_roads_rows)):
        # for pt in xrange(len(vert_roads_rows[road])):
            # terrains[vert_roads_rows[road][pt]]\
                    # [vert_roads_cols[road][pt]] = road

    # 4. randomly place person and tree on the map based on Perlin Noise
    # XXX: we need to make sure that num of person = of tree
    # so we cannot let other element overwrite person nor tree
    # noise will be [-0.,1.]
    noise_values = compute_noise(num_row, num_col, freq=10.)
    persons = []
    for y in xrange(num_row):
        for x in xrange(num_col):
            if noise_values[y][x] >= person_threshold:
                persons.append([y,x])
    trees = []
    noise_values = compute_noise(num_row, num_col, freq=10.)
    for y in xrange(num_row):
        for x in xrange(num_col):
            if noise_values[y][x] >= tree_threshold:
                trees.append([y,x])
    # XXX: we have to make sure the number of trees = of persons
    # so we don't allow person and tree overwrite each other
    for pp in persons:
        for tt in trees:
            if pp == tt:
                persons.remove(pp)
                trees.remove(tt)
    # XXX: we cannot let the human or tree block the road
    persons = [pp for pp in persons if terrains[pp[0]][pp[1]] != road]
    trees = [tt for tt in trees if terrains[tt[0]][tt[1]] != road]
    num_person_tree = min(len(persons),len(trees))
    persons = random.sample(persons,num_person_tree)
    trees = random.sample(trees,num_person_tree)
    assert(len(persons)==len(trees))
    for pp in persons:
        terrains[pp[0]][pp[1]] = person
    for tt in trees:
        terrains[tt[0]][tt[1]] = tree

    return terrains


