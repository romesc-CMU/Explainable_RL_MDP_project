import csv,random
import numpy as np
from PIL import Image

map_numbers = range(100)
batch_dir = './input/batch0/'
file_name_prefix = 'map_0_'


path = './tiles/tileset/tileset11/ground_water/png/separate/256x256/'
# --------------------------------------------------------------
# A. dirt = road
# We will not used 26 30 46 50
# 1. dirt next to 0 grass and 4 dirt
dirt_imgs = [Image.open(path+'dirt/Dirt (1).png'),\
        Image.open(path+'dirt/Dirt (5).png'),\
        Image.open(path+'dirt/Dirt (21).png'),\
        Image.open(path+'dirt/Dirt (25).png'),\
        Image.open(path+'dirt/Dirt (38).png')]

# 2. dirt next to 1 grass and 3 dirt
# dirt with bottom side
dirt_bottom_imgs = [Image.open(path+'dirt/Dirt (2).png'),\
        Image.open(path+'dirt/Dirt (3).png'),\
        Image.open(path+'dirt/Dirt (4).png'),\
        Image.open(path+'dirt/Dirt (43).png')]
# dirt with right side
dirt_right_imgs = [Image.open(path+'dirt/Dirt (6).png'),\
        Image.open(path+'dirt/Dirt (11).png'),\
        Image.open(path+'dirt/Dirt (16).png'),\
        Image.open(path+'dirt/Dirt (39).png')]
# dirt with left side
dirt_left_imgs = [Image.open(path+'dirt/Dirt (10).png'),\
        Image.open(path+'dirt/Dirt (15).png'),\
        Image.open(path+'dirt/Dirt (20).png'),\
        Image.open(path+'dirt/Dirt (37).png')]
# dirt with top side
dirt_top_imgs = [Image.open(path+'dirt/Dirt (22).png'),\
        Image.open(path+'dirt/Dirt (23).png'),\
        Image.open(path+'dirt/Dirt (24).png'),\
        Image.open(path+'dirt/Dirt (33).png')]

# 3. dirt next to 2 asymmetric grass and 2 dirt
# dirt with top side and left side
dirt_top_left_imgs = [Image.open(path+'dirt/Dirt (32).png'),\
        Image.open(path+'dirt/Dirt (74).png'),\
        Image.open(path+'dirt/Dirt (77).png')]
# dirt with top side and right side
dirt_top_right_imgs = [Image.open(path+'dirt/Dirt (34).png'),\
        Image.open(path+'dirt/Dirt (78).png'),\
        Image.open(path+'dirt/Dirt (81).png')]
# dirt with bottom side and left side
dirt_bottom_left_imgs = [Image.open(path+'dirt/Dirt (42).png'),\
        Image.open(path+'dirt/Dirt (75).png'),\
        Image.open(path+'dirt/Dirt (76).png')]
# dirt with bottom side and right side
dirt_bottom_right_imgs = [Image.open(path+'dirt/Dirt (44).png'),\
        Image.open(path+'dirt/Dirt (79).png'),\
        Image.open(path+'dirt/Dirt (80).png')]

# 4. dirt next to 2 symmetric grass and 2 dirt
# dirt with top side and bottom side
dirt_top_bottom_imgs = [Image.open(path+'dirt/Dirt (56).png'),\
        Image.open(path+'dirt/Dirt (57).png'),\
        Image.open(path+'dirt/Dirt (58).png'),\
        Image.open(path+'dirt/Dirt (59).png'),\
        Image.open(path+'dirt/Dirt (60).png'),\
        Image.open(path+'dirt/Dirt (61).png'),\
        Image.open(path+'dirt/Dirt (62).png'),\
        Image.open(path+'dirt/Dirt (63).png'),\
        Image.open(path+'dirt/Dirt (64).png')]
# dirt with left side and right side
dirt_left_right_imgs = [Image.open(path+'dirt/Dirt (65).png'),\
        Image.open(path+'dirt/Dirt (66).png'),\
        Image.open(path+'dirt/Dirt (67).png'),\
        Image.open(path+'dirt/Dirt (68).png'),\
        Image.open(path+'dirt/Dirt (69).png'),\
        Image.open(path+'dirt/Dirt (70).png'),\
        Image.open(path+'dirt/Dirt (71).png'),\
        Image.open(path+'dirt/Dirt (72).png'),\
        Image.open(path+'dirt/Dirt (73).png')]

# 5. dirt next to 3 grass and 1 dirt
# dirt with bottom grass
dirt_top_end_imgs = [Image.open(path+'dirt/Dirt (52).png')]
# dirt with top grass
dirt_bottom_end_imgs = [Image.open(path+'dirt/Dirt (53).png')]
# dirt with right grass
dirt_left_end_imgs = [Image.open(path+'dirt/Dirt (54).png')]
# dirt with left grass
dirt_right_end_imgs = [Image.open(path+'dirt/Dirt (55).png')]

# 6. dirt next to 4 grass and 0 dirt
dirt_4_side_end_imgs = [Image.open(path+'dirt/Dirt (51).png')]

# --------------------------------------------------------------
# B. grass
path = './tiles/tileset/tileset11/ground_water/png/separate/256x256/grass_shen_ps/'
# 1. grass next to 0 dirt and 4 grass
grass_imgs = [Image.open(path+'grass0.png')]
# 2. grass next to 1 dirt and 3 grass
# grass with bottom side
grass_bottom_imgs = [Image.open(path+'grass1.png')]
# grass with right side
grass_right_imgs = [Image.open(path+'grass2.png')]
# grass with left side
grass_left_imgs = [Image.open(path+'grass4.png')]
# grass with top side
grass_top_imgs = [Image.open(path+'grass3.png')]

# 3. grass next to 2 asymmetric dirt and 2 grass
# grass with top side and left side
grass_top_left_imgs = [Image.open(path+'grass9.png')]
# grass with top side and right side
grass_top_right_imgs = [Image.open(path+'grass10.png')]
# grass with bottom side and left side
grass_bottom_left_imgs = [Image.open(path+'grass11.png')]
# grass with bottom side and right side
grass_bottom_right_imgs = [Image.open(path+'grass12.png')]

# 4. grass next to 2 symmetric dirt and 2 grass
# grass with top side and bottom side
grass_top_bottom_imgs = [Image.open(path+'grass5.png')]
# grass with left side and right side
grass_left_right_imgs = [Image.open(path+'grass6.png')]

# 5. grass next to 3 dirt and 1 grass
# grass with bottom grass
grass_top_end_imgs = [Image.open(path+'grass15.png')]
# grass with top grass
grass_bottom_end_imgs = [Image.open(path+'grass13.png')]
# grass with right grass
grass_left_end_imgs = [Image.open(path+'grass7.png')]
# grass with left grass
grass_right_end_imgs = [Image.open(path+'grass14.png')]

# 6. grass next to 4 dirt and 0 grass
grass_4_side_end_imgs = [Image.open(path+'grass8.png')]

# --------------------------------------------------------------
# C. stone
path = './tiles/tileset/tileset11/ground_stone_house/png/2x/stone/'
stone_img = [Image.open(path+'Stone (5).png'), Image.open(path+'Stone (6).png'),\
        Image.open(path+'Stone (7).png'),Image.open(path+'Stone (10).png'),\
        Image.open(path+'Stone (12).png'),Image.open(path+'Stone (14).png'),\
        Image.open(path+'Stone (17).png'),Image.open(path+'Stone (19).png'),\
        Image.open(path+'Stone (21).png'),Image.open(path+'Stone (24).png'),\
        Image.open(path+'Stone (28).png'),Image.open(path+'Stone (32).png'),\
        Image.open(path+'Stone (31).png'),Image.open(path+'Stone (33).png')]

# D. human and tree
path = './tiles/tileset/tileset11/'
human_imgs = [Image.open(path+'Idle (1)_shen_ps.png'), \
        Image.open(path+'Idle__000_shen_ps.png')]
tree_imgs = [Image.open(path+'tree_shen_ps.png')]

# http://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
road = 1
grass = 2
rock = 3
tree = 4
person = 5


for iii in map_numbers:
    # iii = 0
    # terrains = []
    # file_name = file_name_prefix+str(iii)

    # with open(batch_dir+file_name+'.csv', 'rb') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         terrains.append(row)
    # terrains = np.array(terrains)
    # terrains = terrains[:-5]
    terrains = [1,2,3,4,5,6]
    print terrains

    legend_img = []
    for i in xrange(len(terrains)):
        if int(terrains[i]) == tree:
            # we want to first put a grass there, 
            # and then paste the tree image onto it
            # (1) put a grass here
            if i-1>=0 and i-1<len(terrains):
                if int(terrains[i-1]) == grass or \
                        int(terrains[i-1]) == person or \
                        int(terrains[i-1]) == tree:
                    top = True
            if i+1>=0 and i+1<len(terrains):
                if int(terrains[i+1]) == grass or \
                        int(terrains[i+1]) == person or \
                        int(terrains[i+1]) == tree:
                    bottom = True
            # 4 grass next to it 
            if top==True and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(grass_imgs,1)[0]

            # 3 grass next to it 
            elif top==False and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(grass_top_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(grass_bottom_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(grass_left_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(grass_right_imgs,1)[0]

            # 2 grass next to it 
            elif top==False and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(grass_top_bottom_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(grass_top_left_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(grass_top_right_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(grass_bottom_left_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(grass_bottom_right_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(grass_left_right_imgs,1)[0]

            # 1 grass next to it 
            elif top==True and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(grass_bottom_end_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(grass_top_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(grass_right_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(grass_left_end_imgs,1)[0]

            # 0 grass next to it 
            elif top==False and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(grass_4_side_end_imgs,1)[0]
            
            # (2) now we paste the tree onto the grass
            # XXX: we have to copy here
            # otherwise all the grass image - tmp will be changed from no on
            tmp2 = tmp.copy()
            tree_img = random.sample(tree_imgs,1)[0]
            tmp2.paste(tree_img, (0, 0), tree_img)
            legend_img.append(tmp2)

        elif int(terrains[i]) == person:
            # we want to first put a grass there, 
            # and then paste the person image onto it
            # (1)
            top = False
            bottom = False
            left = False
            right = False
            if i-1>=0 and i-1<len(terrains):
                if int(terrains[i-1]) == grass or \
                        int(terrains[i-1]) == person or \
                        int(terrains[i-1]) == tree:
                    top = True
            if i+1>=0 and i+1<len(terrains):
                if int(terrains[i+1]) == grass or \
                        int(terrains[i+1]) == person or \
                        int(terrains[i+1]) == tree:
                    bottom = True

            # 4 grass next to it 
            if top==True and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(grass_imgs,1)[0]

            # 3 grass next to it 
            elif top==False and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(grass_top_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(grass_bottom_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(grass_left_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(grass_right_imgs,1)[0]

            # 2 grass next to it 
            elif top==False and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(grass_top_bottom_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(grass_top_left_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(grass_top_right_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(grass_bottom_left_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(grass_bottom_right_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(grass_left_right_imgs,1)[0]

            # 1 grass next to it 
            elif top==True and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(grass_bottom_end_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(grass_top_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(grass_right_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(grass_left_end_imgs,1)[0]

            # 0 grass next to it 
            elif top==False and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(grass_4_side_end_imgs,1)[0]
            
            # (2) now we paste the person onto the grass
            tmp2 = tmp.copy()
            person_img = human_imgs[0]
            tmp2.paste(person_img, (0, 0), person_img)
            legend_img.append(tmp2)
        elif int(terrains[i]) == 6:
            # we want to first put a grass there, 
            # and then paste the person image onto it
            # (1)
            top = False
            bottom = False
            left = False
            right = False
            if i-1>=0 and i-1<len(terrains):
                if int(terrains[i-1]) == grass or \
                        int(terrains[i-1]) == person or \
                        int(terrains[i-1]) == tree:
                    top = True
            if i+1>=0 and i+1<len(terrains):
                if int(terrains[i+1]) == grass or \
                        int(terrains[i+1]) == person or \
                        int(terrains[i+1]) == tree:
                    bottom = True

            # 4 grass next to it 
            if top==True and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(grass_imgs,1)[0]

            # 3 grass next to it 
            elif top==False and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(grass_top_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(grass_bottom_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(grass_left_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(grass_right_imgs,1)[0]

            # 2 grass next to it 
            elif top==False and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(grass_top_bottom_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(grass_top_left_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(grass_top_right_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(grass_bottom_left_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(grass_bottom_right_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(grass_left_right_imgs,1)[0]

            # 1 grass next to it 
            elif top==True and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(grass_bottom_end_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(grass_top_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(grass_right_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(grass_left_end_imgs,1)[0]

            # 0 grass next to it 
            elif top==False and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(grass_4_side_end_imgs,1)[0]
            
            # (2) now we paste the person onto the grass
            tmp2 = tmp.copy()
            person_img = human_imgs[1]
            tmp2.paste(person_img, (0, 0), person_img)
            legend_img.append(tmp2)
        elif int(terrains[i]) == road:
            top = False
            bottom = False
            left = False
            right = False
            if i-1>=0 and i-1<len(terrains):
                if int(terrains[i-1]) == road:
                    top = True
            if i+1>=0 and i+1<len(terrains):
                if int(terrains[i+1]) == road:
                    bottom = True
            # 4 road next to it 
            if top==True and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(dirt_imgs,1)[0]

            # 3 road next to it 
            elif top==False and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(dirt_top_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(dirt_bottom_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(dirt_left_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(dirt_right_imgs,1)[0]

            # 2 road next to it 
            elif top==False and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(dirt_top_bottom_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(dirt_top_left_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(dirt_top_right_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(dirt_bottom_left_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(dirt_bottom_right_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(dirt_left_right_imgs,1)[0]

            # 1 road next to it 
            elif top==True and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(dirt_bottom_end_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(dirt_top_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(dirt_right_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(dirt_left_end_imgs,1)[0]

            # 0 road next to it 
            elif top==False and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(dirt_4_side_end_imgs,1)[0]
            legend_img.append(tmp)

        elif int(terrains[i]) == rock:
            legend_img.append(random.sample(stone_img,1)[0])

        elif int(terrains[i]) == grass:
            top = False
            bottom = False
            left = False
            right = False
            if i-1>=0 and i-1<len(terrains):
                if int(terrains[i-1]) == grass or \
                        int(terrains[i-1]) == person or \
                        int(terrains[i-1]) == tree:
                    top = True
            if i+1>=0 and i+1<len(terrains):
                if int(terrains[i+1]) == grass or \
                        int(terrains[i+1]) == person or \
                        int(terrains[i+1]) == tree:
                    bottom = True

            # 4 grass next to it 
            if top==True and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(grass_imgs,1)[0]

            # 3 grass next to it 
            elif top==False and bottom==True \
                    and left==True and right==True:
                tmp = random.sample(grass_top_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(grass_bottom_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(grass_left_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(grass_right_imgs,1)[0]

            # 2 grass next to it 
            elif top==False and bottom==False \
                    and left==True and right==True:
                tmp = random.sample(grass_top_bottom_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==True:
                tmp = random.sample(grass_top_left_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==True and right==False:
                tmp = random.sample(grass_top_right_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(grass_bottom_left_imgs,1)[0]
            elif top==True and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(grass_bottom_right_imgs,1)[0]
            elif top==True and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(grass_left_right_imgs,1)[0]

            # 1 grass next to it 
            elif top==True and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(grass_bottom_end_imgs,1)[0]
            elif top==False and bottom==True \
                    and left==False and right==False:
                tmp = random.sample(grass_top_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==True and right==False:
                tmp = random.sample(grass_right_end_imgs,1)[0]
            elif top==False and bottom==False \
                    and left==False and right==True:
                tmp = random.sample(grass_left_end_imgs,1)[0]

            # 0 grass next to it 
            elif top==False and bottom==False \
                    and left==False and right==False:
                tmp = random.sample(grass_4_side_end_imgs,1)[0]
            legend_img.append(tmp)
        else:
            # legend_img.append(ground_img)
            print "WTF"
            exit(0)
    # print legend_img


min_shape = sorted( [(np.sum(i.size), i.size ) for i in legend_img])[0][1]
imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in legend_img))
imgs_comb = Image.fromarray(imgs_comb)
imgs_comb.save('./legend.png')