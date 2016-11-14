from __future__ import division 
from solve_mdp.solve_mdp import generate_path,write_file
from solve_mdp.draw_path import draw_path,draw_start_end
import csv,sys,random,os,math,copy
import numpy as np
from sets import Set
from IPython import embed


if __name__ == "__main__":
    # You don't need to change
    filename_prefix = 'map_'
    # input_directory = './input/noise_based_10_20/batch0/'
    # output_directory = './output/noise_based_10_20/batch0/'
    input_directory = './output_maps/'
    output_directory = './output_maps/'

    # Select the maps you want to plan optimal paths on
    # >> in ./input/noise_based_10_20/batch0/, there are 100 maps
    # >> so to generate the paths for all the maps, then you should do:
    # >> image_list = np.arange(0,100)
    image_list = [25]
    #image_list = [0]

    # change reward
    reward_functions = []
    #                   (index, (road, rock, grass, human, tree, goal))
    reward_functions.append((0, (-1000.,-1.,-1.,-1000.,-1000.,100.),))
    reward_functions.append((1, (-1000,-3.,-1.,-1000.,-1000.,100.),))
    reward_functions.append((2, (-1000.,-1.,-3.,-1000.,-1000.,100.),))
    #reward_functions.append((3, (-20.,-1000.,-1.,-1000.,-1000.,100.),))
    #reward_functions.append((4, (-1.,-1000.,-20.,-1000.,-1000.,100.),))

    # change start and end point
    # >> you need to make sure you are not starting nor ending on a person or a tree
    # >> if you don't assign them, they will be generated randomly
    starting_pt_row = 2
    starting_pt_col = 0
    ending_pt_row = 2
    ending_pt_col = 19

    # ---------------------------------------------------------------
    # 1. plan paths
    # XXX: using MDP. Since we don't have discount, 
    # the only way to make value iteration converge is by using absorbing state. 
    # Therefore, if the rewards are not good enough, value iteration will not converge
    # try:
    # >> if you don't assign start or end, they will be generated randomly
    # >> if you don't want to eliminate duplicates, pass False in the last argument
    # benchmarks = test_case(reward_functions, image_list, \
            # filename_prefix, input_directory, output_directory,\
            # eliminate_duplicate=True)
    benchmarks = generate_path(reward_functions, image_list, \
            filename_prefix, input_directory, output_directory, \
            starting_pt_row, starting_pt_col,\
            ending_pt_row, ending_pt_col, eliminate_duplicate=True)
    # except:
        # print("Unexpected error:", sys.exc_info()[0])
    

    # ---------------------------------------------------------------
    # 2. save the data in ./output/noise_based_10_20/batch0/
    write_file(benchmarks)

    # ---------------------------------------------------------------
    # 3. draw empty maps with only start and point and save them in ./output/noise_based_10_20/batch0/
    # they are named as "map_0_1_-1.png"
    draw_start_end(benchmarks)
 
    # ---------------------------------------------------------------
    # 4. draw paths on the images and save them in ./output/noise_based_10_20/batch0/
    draw_path(benchmarks)

 





