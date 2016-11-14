from __future__ import division
import random,math
import numpy as np
# from matplotlib import pyplot as plt

from bezier_curve import bezier_curve

# http://www.redblobgames.com/articles/curved-paths/
def generate_bezier_road(num_row, num_col, num_pt_bezier=4, horizontal=True):
    xvals, yvals = generate_bezier_curve(num_row, num_col, num_pt_bezier, horizontal)
    # pts are in (x,y)
    pts_rows = []
    pts_cols = []
    for i in xrange(len(xvals)):
        pt = [xvals[i],yvals[i]]
        pts_rows.append(int(round(pt[1]))) 
        pts_cols.append(int(round(pt[0])))
    # plt.plot(pts_cols, pts_rows)
    # plt.show()

    return pts_rows, pts_cols


def generate_bezier_curve(num_row, num_col, nPoints=4, horizontal=True):
    assert(nPoints>=2)
    random.seed()

    if horizontal==True:
        # points in (x,y)
        points = [[0, random.uniform(0,num_row-1)]]
        for i in xrange(nPoints-2):
            points.append([random.uniform(0,num_col-1), random.uniform(0,num_row-1)])
        points.append([num_col-1, random.uniform(0,num_row-1)])
    else:
        # points in (x,y)
        points = [[random.uniform(0,num_col-1), 0]]
        for i in xrange(nPoints-2):
            points.append([random.uniform(0,num_col-1), random.uniform(0,num_row-1)])
        points.append([random.uniform(0,num_col-1), num_row-1])

    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]

    xvals, yvals = bezier_curve(points, nTimes=1000)
    # plt.plot(xvals, yvals)
    # plt.plot(xpoints, ypoints, "ro")
    # for nr in range(len(points)):
        # plt.text(points[nr][0], points[nr][1], nr)
    # plt.show()
    return xvals, yvals

def in_bound(i,j,num_row,num_col):
    if i >= 0 and i < num_row and j >= 0 and j < num_col:
        return True
    else:
        return False


def generate_horizontal_road(num_row, num_col, \
    opts_history_window_size=3, opts_history_prob_param=0.5):
    random.seed()
    i0 = random.randint(0,num_row-1)
    road_rows = [i0]
    road_cols = [0]
    next_opts = ['up','down','right']
    road_opts = []
    for j in xrange(1,num_col):
        opt = random.sample(next_opts,1)[0]
        # we want it to follow the closest history
        opts_history = []
        if len(road_opts) >= opts_history_window_size:
            opts_history = road_opts[-opts_history_window_size:]
        else:
            opts_history = road_opts
        counts = [opts_history.count(x) for x in next_opts]
        if sum(counts) == 0:
            probabilities_norm = [1./len(counts) for _ in counts]
            # print opts_history 
            # print probabilities_norm 
            # print '---------------'
        else:
            # to avoid prob == 0, we use this
            probabilities = [math.pow(opts_history_prob_param*\
                    math.e, x)/sum(counts) for x in counts]
            # probabilities = [x/sum(counts) for x in counts]
            probabilities_norm = [x/sum(probabilities) for x in probabilities]
            # print probabilities_norm 
            # print opts_history 
            # print '---------------'

        opt = np.random.choice(np.arange(0, len(next_opts)), \
                p=probabilities_norm)
        opt = next_opts[opt]
        road_opts.append(opt)

        if opt == 'up':
            # XXX: the point ([-1],j) must be added before ([-1]-1,j)
            if in_bound(road_rows[-1],j,num_row,num_col):
                road_rows.append(road_rows[-1])
                road_cols.append(j)
            if in_bound(road_rows[-1]-1,j,num_row,num_col):
                road_rows.append(road_rows[-1]-1)
                road_cols.append(j)
            next_opts = ['up','right']

        elif opt == 'down':
            if in_bound(road_rows[-1],j,num_row,num_col):
                road_rows.append(road_rows[-1])
                road_cols.append(j)
            if in_bound(road_rows[-1]+1,j,num_row,num_col):
                road_rows.append(road_rows[-1]+1)
                road_cols.append(j)
            next_opts = ['down','right']
        else:
            if in_bound(road_rows[-1],j,num_row,num_col):
                road_rows.append(road_rows[-1])
                road_cols.append(j)
            next_opts = ['up','down','right']
    return road_rows, road_cols

def generate_vertical_road(num_row, num_col, \
    opts_history_window_size=3, opts_history_prob_param=0.5):
    random.seed()
    i0 = random.randint(0,num_col-1)
    road_rows = [0]
    road_cols = [i0]
    next_opts = ['left','right','down']
    road_opts = []
    for i in xrange(1,num_row):
        opt = random.sample(next_opts,1)[0]
        # we want it to follow the closest history
        opts_history = []
        if len(road_opts) >= opts_history_window_size:
            opts_history = road_opts[-opts_history_window_size:]
        else:
            opts_history = road_opts
        counts = [opts_history.count(x) for x in next_opts]
        if sum(counts) == 0:
            probabilities_norm = [1./len(counts) for _ in counts]
            # print opts_history 
            # print probabilities_norm 
            # print '---------------'
        else:
            # to avoid prob == 0, we use this
            probabilities = [math.pow(opts_history_prob_param*\
                    math.e, x)/sum(counts) for x in counts]
            # probabilities = [x/sum(counts) for x in counts]
            probabilities_norm = [x/sum(probabilities) for x in probabilities]
            # print probabilities_norm 
            # print opts_history 
            # print '---------------'

        opt = np.random.choice(np.arange(0, len(next_opts)), \
                p=probabilities_norm)
        opt = next_opts[opt]
        road_opts.append(opt)

        if opt == 'left':
            if in_bound(i,road_cols[-1],num_row,num_col):
                road_rows.append(i)
                road_cols.append(road_cols[-1])
            if in_bound(i,road_cols[-1]-1,num_row,num_col):
                road_rows.append(i)
                road_cols.append(road_cols[-1]-1)
            next_opts = ['left','down']

        elif opt == 'right':
            if in_bound(i,road_cols[-1],num_row,num_col):
                road_rows.append(i)
                road_cols.append(road_cols[-1])
            if in_bound(i,road_cols[-1]+1,num_row,num_col):
                road_rows.append(i)
                road_cols.append(road_cols[-1]+1)
            next_opts = ['right','down']
        else:
            if in_bound(i,road_cols[-1],num_row,num_col):
                road_rows.append(i)
                road_cols.append(road_cols[-1])
            next_opts = ['left','right','down']
    return road_rows, road_cols

