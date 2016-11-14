# from build_terrain_random.generate_terrain import generate_terrain
from build_terrain_noise_based.generate_terrain import generate_terrain
from draw_map.draw_tilemap import draw_tilemap
import os,csv
from PIL import Image
from IPython import embed

road = 1
grass = 2
rock = 3
tree = 4
person = 5


def generate_map():
    num_row = 10
    num_col = 20

    # person_threshold = 0.68
    # tree_threshold = 0.7
    # rock_threshold = 0.55
    # num_vertical_road = 0
    # num_horizontal_road = 1

    # the higher, the less randomly generated
    person_threshold = 1.
    # the higher, the less randomly generated
    tree_threshold = 1.
    # the higher, the less randomly generated
    rock_threshold = .5
    num_vertical_road = 0
    num_horizontal_road = 0 
    background = grass

    map_number = 25 

    output_directory = './output_maps/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    file_prefix = output_directory+'map_'+str(map_number)

    terrains = generate_terrain(num_row=num_row, num_col=num_col,\
            person_threshold=person_threshold,\
            tree_threshold=tree_threshold,\
            rock_threshold=rock_threshold,\
            num_horizontal_road=num_horizontal_road,\
            num_vertical_road=num_vertical_road,\
            background=background)

    with open(file_prefix+'.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for t in (terrains):
            spamwriter.writerow(t)

    tilemap = draw_tilemap(terrains)
    tilemap.save(file_prefix+'.png')







if __name__ == "__main__":
    generate_map()

