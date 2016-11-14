from __future__ import division 
from mdp import mdp_path_generation
import csv,sys,random,os,math,copy
import numpy as np
from sets import Set
from IPython import embed
from draw_map.draw_tilemap import draw_tilemap
from PIL import Image


road = 1
grass = 2
rock = 3
tree = 4
person = 5


def generate_path(reward_functions, image_list, filename_prefix,\
        input_directory_, output_directory_,\
        starting_pt_row=-1, starting_pt_col=-1,\
        ending_pt_row=-1, ending_pt_col=-1, eliminate_duplicate=True):

    random.seed()

    paths = []
    benchmarks = []
    for _ in image_list:
        paths.append([])
        benchmarks.append([])
 
    for idx, image_index in enumerate(image_list):
        # 1. read the terrains
        input_filename = filename_prefix+str(image_index)
        # print 'input_filename=', input_filename
        input_directory = input_directory_+str(input_filename)
        terrains = []
        with open(input_directory+'.csv', 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                terrains.append(row)
        for y in xrange(len(terrains)):
            for x in xrange(len(terrains[0])):
                terrains[y][x] = int(terrains[y][x])

        # 2. randomly choose the start and end point
        if starting_pt_row==-1 or starting_pt_col==-1 or \
                ending_pt_row==-1 or ending_pt_col==-1:
            # in this case, we go from the left end to the right end
            while True:
                starting_pt_row = random.randint(round(len(terrains)/3.)-1,\
                        round(len(terrains)/3.*2.+1))
                starting_pt_col = 0
                ending_pt_row = starting_pt_row
                ending_pt_col = len(terrains[0])-1
                if terrains[starting_pt_row][starting_pt_col] != person \
                        and terrains[starting_pt_row][starting_pt_col] != tree \
                        and terrains[ending_pt_row][ending_pt_col] != person \
                        and terrains[ending_pt_row][ending_pt_col] != tree:
                    break
        # print 'start at row=', starting_pt_row, ', col=', starting_pt_col
        # print 'end at row=', ending_pt_row, ', col=', ending_pt_col

        # make sure each <map,rfc> has only one path
        while True:
            paths_per_map= []
            for i in range(len(reward_functions)):
                paths_per_map.append([])
            benchmarks_per_map = []
            for i in range(len(reward_functions)):
                benchmarks_per_map.append([])

            for reward_function in reward_functions:
                rf_index = reward_function[0]
                road_reward = reward_function[1][0]
                rock_reward = reward_function[1][1]
                grass_reward = reward_function[1][2]
                human_reward = reward_function[1][3]
                tree_reward = reward_function[1][4]
                goal_reward = reward_function[1][5]
                output_directory = output_directory_+str(input_filename)+'/'\
                        +input_filename+'_rf'+str(rf_index)

                # 3. plan
                # (1) compute reward based on the benchmark
                reward_map = []
                for i in xrange(len(terrains)):
                    row_reward = []
                    for j in xrange(len(terrains[0])):
                        e = terrains[i][j]
                        terrain_reward = 0.
                        if int(e) == road:
                            terrain_reward = road_reward
                        elif int(e) == tree:
                            terrain_reward = tree_reward
                        elif int(e) == person:
                            terrain_reward = human_reward
                        elif int(e) == rock:
                            terrain_reward = rock_reward
                        elif int(e) == grass:
                            terrain_reward = grass_reward
                        elif int(e) == 0:
                            print "WTF"
                            exit(0)
                        row_reward.append(terrain_reward)
                    reward_map.append(row_reward)
                reward_map[ending_pt_row][ending_pt_col] = goal_reward


                # (2) rank the tiles which really exists in the map
                element_in_terrains = {}
                for t in terrains:
                    for tt in t:
                        pair = ()
                        if tt == grass:
                            element_in_terrains[tt] = grass_reward
                        elif tt == rock:
                            element_in_terrains[tt] = rock_reward
                        elif tt == road:
                            element_in_terrains[tt] = road_reward
                        elif tt == person:
                            element_in_terrains[tt] = human_reward
                        elif tt == tree:
                            element_in_terrains[tt] = tree_reward
                        else:
                            print 'wtf'
                            exit(0)

                # (3) MDP
                # XXX: grid_map, terminals, init, arrow_map, are all in [row,col]
                # XXX: The Python Imaging Library uses a Cartesian pixel coordinate system
                # with (0,0) in the upper left corner.
                # draw.text((x,y)...)
                # XXX: The MDP value iteration allows multiple solutions
                # gamma = 1. 
                # epsilon = 0.001
                utility_map, opt_action_map, paths_per_map_per_rfct = \
                        mdp_path_generation(\
                        reward_map = reward_map, \
                        terminals = [(ending_pt_row, ending_pt_col)],\
                        starting_pt_row = starting_pt_row,\
                        starting_pt_col = starting_pt_col,\
                        ending_pt_row = ending_pt_row,\
                        ending_pt_col = ending_pt_col)
                # print len(paths_per_map_per_rfct)
                assert(len(set(paths_per_map_per_rfct))==len(paths_per_map_per_rfct))
                for p in paths_per_map_per_rfct:
                    assert(len(p)>=20)
                    assert(len(p)==len(set(p)))
                paths_per_map[rf_index] = paths_per_map_per_rfct
                print str(len(paths_per_map_per_rfct))+" paths"

                # (4) save benchmark data
                benchmark_data = [rf_index, road_reward, rock_reward, \
                        grass_reward, human_reward,\
                        tree_reward, goal_reward,\
                        input_directory, output_directory,\
                        starting_pt_row, starting_pt_col,\
                        ending_pt_row, ending_pt_col]
                benchmark_data_header = ['rf_index', 'road_reward', 'rock_reward', \
                        'grass_reward', 'human_reward', \
                        'tree_reward', 'goal_reward',\
                        'input_directory', 'output_directory',\
                        'starting_pt_row', 'starting_pt_col',\
                        'ending_pt_row', 'ending_pt_col']
                benchmarks_per_map[rf_index] = {
                    'reward_map': reward_map,
                    'reward_function': reward_function,
                    'opt_action_map': opt_action_map,
                    'utility_map': utility_map,
                    'paths_per_map_per_rfct': paths_per_map_per_rfct,
                    'benchmark_data': benchmark_data,
                    'benchmark_data_header': benchmark_data_header,
                    'input_directory': input_directory,
                    'output_directory': output_directory,
                    'input_filename': input_filename,
                    }

            # (4) Now we will try to change the map to eliminatee other paths
            # we count how many times a point appears among all the possible paths
            # make sure each <map,rfc> has only one path
            if eliminate_duplicate==True:
                if all(i == 1 for i in [len(x) for x in paths_per_map]):
                    break
                num_occur_per_map = {}
                for paths_per_map_per_rfct in paths_per_map:
                    if len(paths_per_map_per_rfct) > 0:
                        num_occur_per_map_per_rfct = {}
                        for p in paths_per_map_per_rfct:
                            for pp in p:
                                if pp not in num_occur_per_map_per_rfct:
                                    num_occur_per_map_per_rfct[pp] = 0
                                num_occur_per_map_per_rfct[pp] += 1
                        num_path = len(paths_per_map_per_rfct)
                        num_occur_per_map_per_rfct = {k:v for (k,v) in \
                                num_occur_per_map_per_rfct.iteritems() if v < num_path}
                        for k,v in num_occur_per_map_per_rfct.iteritems():
                            if k not in num_occur_per_map:
                                num_occur_per_map[k] = v
                            else:
                                num_occur_per_map[k] += v
                found = False 
                # a. we find the most frequently appeared tile and increase the cost
                num_occur_per_map_1 = copy.deepcopy(num_occur_per_map)
                while len(num_occur_per_map_1)>0:
                    max_act = max(num_occur_per_map_1.items(), key=lambda x: x[1])
                    max_act_reward = element_in_terrains[terrains[max_act[0][0]][max_act[0][1]]]
                    kv_tmp = []
                    for k,v in element_in_terrains.iteritems():
                        if v < max_act_reward:
                            kv_tmp.append((k,v))
                    if len(kv_tmp)>0:
                        found = True
                        # XXX: we randomly choose one tile to change to
                        tmp = random.sample(kv_tmp, 1)[0]
                        v = tmp[1]
                        k = tmp[0]
                        reward_map[max_act[0][0]][max_act[0][1]] = v
                        terrains[max_act[0][0]][max_act[0][1]] = k
                        print 'we change (',max_act[0][0],', ',max_act[0][1],') to '+str(k)
                        break
                    if found == False:
                        del num_occur_per_map_1[max_act[0]]
                    else:
                        break
                if found == False:
                    # b. we find the least frequently appeared tile and decrease the cost
                    # Rosario - changed this from num_occur_per_map_2 to 1
                    num_occur_per_map_2 = copy.deepcopy(num_occur_per_map_1)
                    while len(num_occur_per_map_2)>0:
                        min_act = min(num_occur_per_map_2.items(), key=lambda x: x[1])
                        min_act_reward = element_in_terrains[terrains[min_act[0][0]][min_act[0][1]]]
                        for k,v in element_in_terrains.iteritems():
                            if v > min_act_reward:
                                found = True
                                reward_map[min_act[0][0]][min_act[0][1]] = v
                                terrains[min_act[0][0]][min_act[0][1]] = k
                                print 'we change (',min_act[0][0],', ',min_act[0][1],') to '+str(k)
                                break
                        if found == False:
                            del num_occur_per_map_2[min_act[0]]
                        else:
                            break
         
                    if found == False:
                        print 'I cannot remove duplicate'
                    break
            else:
                break

        # Now ideally, we have a map with 1 unique path for each rfct
        paths[idx] = paths_per_map
        benchmarks[idx] = benchmarks_per_map

        # (5) redraw map
        # in order to eliminate duplicate paths, we might need to change the map
        file_prefix = input_directory
        with open(file_prefix+'.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            for t in (terrains):
                spamwriter.writerow(t)
        tilemap = draw_tilemap(terrains)
        tilemap.save(file_prefix+'.png')
    return benchmarks


def write_file(benchmarks):
    for benchmarks_per_map in benchmarks:
        for benchmarks_per_map_per_rfct in benchmarks_per_map:
            reward_map = benchmarks_per_map_per_rfct['reward_map']
            utility_map = benchmarks_per_map_per_rfct['utility_map']
            output_directory = benchmarks_per_map_per_rfct['output_directory']
            benchmark_data_header = benchmarks_per_map_per_rfct['benchmark_data_header']
            benchmark_data = benchmarks_per_map_per_rfct['benchmark_data']
            paths_per_map_per_rfct = benchmarks_per_map_per_rfct['paths_per_map_per_rfct']
            input_directory = benchmarks_per_map_per_rfct['input_directory']
            opt_action_map = benchmarks_per_map_per_rfct['opt_action_map']
            input_filename = benchmarks_per_map_per_rfct['input_filename']
            reward_function = benchmarks_per_map_per_rfct['reward_function']
            rf_index = reward_function[0]

            output_directory = output_directory.split('/')[0]+'/'\
                    +output_directory.split('/')[1]+'/'\
                    +output_directory.split('/')[2]
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_directory += '/'+input_filename+'_rf_'+str(rf_index)

            with open(output_directory+'_reward_map.csv','wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                for row in reward_map:
                    spamwriter.writerow(row)

            with open(output_directory+'_arrow_map.csv','wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                for row in opt_action_map:
                    spamwriter.writerow(row)

            with open(output_directory+'_utility_map.csv','wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                for row in utility_map:
                    spamwriter.writerow(row)

            for i in xrange(len(paths_per_map_per_rfct)):
                with open(output_directory+'_sol_'+str(i)+'_path_row_col.csv','wb') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',')
                    for pt in paths_per_map_per_rfct[i]:
                        spamwriter.writerow(pt)

            with open(output_directory+'_param.csv','wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow(benchmark_data_header)
                spamwriter.writerow(benchmark_data)



# if __name__ == "__main__":
    # filename_prefix = 'map_0_'
    # image_list = np.arange(0,100)
    # image_list = [0]

    # road_reward = -1
    # rock_reward = -1
    # grass_reward = -1
    # human_reward = -10000
    # tree_reward = -1000
    # goal_reward = 100
    # starting_pt_row = 5
    # starting_pt_col = 0
    # ending_pt_row = 5
    # ending_pt_col = 19

    # input_directory = './input/noise_based_10_20/batch0/'
    # output_directory = './output/noise_based_10_20/batch0/'
    # rf_index = 0
    # # try:
    # # benchmarks = test_case(road_reward, rock_reward, grass_reward, \
            # # human_reward, tree_reward, goal_reward, image_list, \
            # # filename_prefix, input_directory, output_directory, rf_index)
    # benchmarks = test_case(road_reward, rock_reward, grass_reward, \
            # human_reward, tree_reward, goal_reward, image_list, \
            # filename_prefix, input_directory, output_directory, \
            # starting_pt_row, starting_pt_col,\
            # ending_pt_row, ending_pt_col, rf_index)
    # # except:
        # # print("Unexpected error:", sys.exc_info()[0])

    # write_file(benchmarks)
    # embed()
 





