import csv,random
import numpy as np

# http://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
road = 1
grass = 2
rock = 3
tree = 4
person = 5

for iii in xrange(1):
    if 1==1:
        iii=1
    terrains = []
    batch_dir = './input/'
    file_name = 'map_0_'+str(iii)

    terrains = []
    with open(batch_dir+file_name+'.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            terrains.append(row)
    terrains = np.array(terrains)
    terrains = terrains[:-3]
    print terrains

    from PIL import Image
    # for road in any conditions, we use circles so that it looks nicer
    bad_road_img = Image.open('./tiles/bad_road.png')
    forest_img = Image.open('./tiles/forest.png')
    tree_img = Image.open('./tiles/tree_on_grass.png')
    grass_img = Image.open('./tiles/grass.png')
    ground_img = Image.open('./tiles/ground.png')
    house1_img = Image.open('./tiles/house1.png')
    house2_img = Image.open('./tiles/house2.png')
    house3_img = Image.open('./tiles/house3.png')
    house4_img = Image.open('./tiles/house4.png')
    human_img = Image.open('./tiles/human2_on_grass.png')
    ice_img = Image.open('./tiles/ice.png')
    road_img = Image.open('./tiles/road.png')
    rock_img = Image.open('./tiles/rock.png')
    sand_img = Image.open('./tiles/sand.png')
    traffic_img = Image.open('./tiles/traffic.png')
    water_img = Image.open('./tiles/water.png')

    terrains_img = []
    for row in terrains:
        row_img = []
        for e in row:
            if int(e) == tree:
                row_img.append(tree_img)
            elif int(e) == person:
                row_img.append(human_img)
            elif int(e) == road:
                row_img.append(road_img)
            elif int(e) == rock:
                row_img.append(bad_road_img)
            elif int(e) == grass:
                row_img.append(ground_img)
            else:
                row_img.append(sand_img)

        min_shape = sorted( [(np.sum(i.size), i.size ) for i in row_img])[0][1]
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in row_img))
        imgs_comb = Image.fromarray(imgs_comb)
        terrains_img.append(imgs_comb)

    min_shape = sorted( [(np.sum(i.size), i.size ) for i in terrains_img])[0][1]
    imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in terrains_img))
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(batch_dir+file_name+'_simple.png')    

