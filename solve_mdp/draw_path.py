from __future__ import division 
import os,csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from IPython import embed

def draw_path(benchmarks):
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
 
            for i in xrange(len(paths_per_map_per_rfct)):
                path = paths_per_map_per_rfct[i]
                output_filename = output_directory+'/'+input_filename+'_rf_'+\
                        str(rf_index)+'_sol'+str(i)
                # print output_filename

                # get the information from image
                image_filename = input_directory+".png"
                im = Image.open(image_filename)
                image_width, image_height = im.size
                grid_width = image_width/len(reward_map[0])
                grid_height = image_height/len(reward_map)

                starting_pt_row = path[0][0]
                starting_pt_col = path[0][1]
                ending_pt_row = path[-1][0]
                ending_pt_col = path[-1][1]

                # draw
                image_filename = input_directory+".png"
                im = Image.open(image_filename)

                # XXX: draw triangles (not good because it indicates 
                # that our robot will turn but it actually doesn't
                """
                draw_pts = []
                for [i,j] in path_row_col:
                    i = int(i)
                    j = int(j)
                    center_x = (j+0.5)*grid_width
                    center_y = (i+0.5)*grid_height
                    if arrow_map[i][j] == '>':
                        draw_pts.append([(center_x+0.2*grid_width, center_y),\
                                (center_x,center_y-0.2*grid_height),\
                                (center_x,center_y+0.2*grid_height)])
                    elif arrow_map[i][j] == '<':
                        draw_pts.append([(center_x-0.2*grid_width, center_y),\
                                (center_x,center_y-0.2*grid_height),\
                                (center_x,center_y+0.2*grid_height)])
                    elif arrow_map[i][j] == '^':
                        draw_pts.append([(center_x,center_y-0.2*grid_height),\
                                (center_x-0.2*grid_width,center_y),\
                                (center_x+0.2*grid_width,center_y)])
                    elif arrow_map[i][j] == 'v':
                        draw_pts.append([(center_x,center_y+0.2*grid_height),\
                                (center_x-0.2*grid_width,center_y),\
                                (center_x+0.2*grid_width,center_y)])
                for triangle in draw_pts:
                    draw = ImageDraw.Draw(im)
                    draw.polygon(triangle, fill='red', outline='red')
                    del draw
                """

                draw_pts = []
                for (i,j) in path:
                    i = int(i)
                    j = int(j)
                    center_x = (j+0.5)*grid_width
                    center_y = (i+0.5)*grid_height
                    draw_pts.append([center_x,center_y])
                assert(len(draw_pts)==len(path))
                # print draw_pts

                # XXX: we draw lines to show trajectory
                """
                for ii in xrange(len(draw_pts)-1):
                    draw = ImageDraw.Draw(im)
                    draw.line([(draw_pts[ii][0],draw_pts[ii][1]),\
                        (draw_pts[ii+1][0],draw_pts[ii+1][1])], width=40,fill='red')
                    del draw
                """

                # XXX: we draw circles to show trajectory
                for ii in xrange(len(draw_pts)):
                    draw = ImageDraw.Draw(im)
                    ellipse_radii = 0.25
                    draw.ellipse([(draw_pts[ii][0]-ellipse_radii*grid_width,\
                            draw_pts[ii][1]-ellipse_radii*grid_height),\
                            (draw_pts[ii][0]+ellipse_radii*grid_height,\
                            draw_pts[ii][1]+ellipse_radii*grid_height)], fill='black')
                    del draw

                # XXX: if there is a point right above the start point,
                # we write 'start' under it
                draw = ImageDraw.Draw(im)
                font = ImageFont.truetype("./solve_mdp/merriweather/Merriweather UltraBold.ttf", 150)
                draw.text(((starting_pt_col+0.1)*grid_width,\
                        (starting_pt_row+0.3)*grid_height), 'start', fill='blue', font=font)
                del draw
                draw = ImageDraw.Draw(im)
                font = ImageFont.truetype("./solve_mdp/merriweather/Merriweather UltraBold.ttf", 150)
                draw.text(((ending_pt_col-0.4)*grid_width,\
                        (ending_pt_row+0.3)*grid_height), 'goal', fill='blue', font=font)
                del draw


                # down sample it
                w = 800
                h = 400
                # http://stackoverflow.com/questions/7936154/python-image-library-clean-downsampling
                if 'P' in im.mode: # check if image is a palette type
                    im = im.convert("RGB") # convert it to RGB
                    im = im.resize((w,h),Image.ANTIALIAS) # resize it
                    im = im.convert("P",dither=Image.NONE, palette=Image.ADAPTIVE) 
                    #convert back to palette
                else:
                    im = im.resize((w,h),Image.ANTIALIAS) # regular resize
                # im.save(newSourceFile) # save the image to the new source
                # im.save(newSourceFile, quality = 95, dpi=(72,72), optimize = True) 
                # set quality, dpi , and shrink size

                im.save(output_filename+'.png')
                print output_filename+'.png'


# draw empty path
def draw_start_end(benchmarks):
    for benchmarks_per_map in benchmarks:
        benchmarks_per_map_per_rfct = benchmarks_per_map[0]
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

        path = paths_per_map_per_rfct[0]
        for i in xrange(len(paths_per_map_per_rfct)):
            assert(paths_per_map_per_rfct[i][0][0]==paths_per_map_per_rfct[0][0][0])
            assert(paths_per_map_per_rfct[i][0][1]==paths_per_map_per_rfct[0][0][1])

        output_directory = output_directory.split('/')[0]+'/'\
                +output_directory.split('/')[1]+'/'\
                +output_directory.split('/')[2]
        output_filename = output_directory+'/'+input_filename+'_rf_-1'
        # print output_filename

        # get the information from image
        image_filename = input_directory+".png"
        im = Image.open(image_filename)
        image_width, image_height = im.size
        grid_width = image_width/len(reward_map[0])
        grid_height = image_height/len(reward_map)

        starting_pt_row = path[0][0]
        starting_pt_col = path[0][1]
        ending_pt_row = path[-1][0]
        ending_pt_col = path[-1][1]

        # draw
        image_filename = input_directory+".png"
        im = Image.open(image_filename)

        # XXX: draw triangles (not good because it indicates 
        # that our robot will turn but it actually doesn't
        """
        draw_pts = []
        for [i,j] in path_row_col:
            i = int(i)
            j = int(j)
            center_x = (j+0.5)*grid_width
            center_y = (i+0.5)*grid_height
            if arrow_map[i][j] == '>':
                draw_pts.append([(center_x+0.2*grid_width, center_y),\
                        (center_x,center_y-0.2*grid_height),\
                        (center_x,center_y+0.2*grid_height)])
            elif arrow_map[i][j] == '<':
                draw_pts.append([(center_x-0.2*grid_width, center_y),\
                        (center_x,center_y-0.2*grid_height),\
                        (center_x,center_y+0.2*grid_height)])
            elif arrow_map[i][j] == '^':
                draw_pts.append([(center_x,center_y-0.2*grid_height),\
                        (center_x-0.2*grid_width,center_y),\
                        (center_x+0.2*grid_width,center_y)])
            elif arrow_map[i][j] == 'v':
                draw_pts.append([(center_x,center_y+0.2*grid_height),\
                        (center_x-0.2*grid_width,center_y),\
                        (center_x+0.2*grid_width,center_y)])
        for triangle in draw_pts:
            draw = ImageDraw.Draw(im)
            draw.polygon(triangle, fill='red', outline='red')
            del draw
        """

        path = [path[0],path[-1]]
        draw_pts = []
        for (i,j) in path:
            i = int(i)
            j = int(j)
            center_x = (j+0.5)*grid_width
            center_y = (i+0.5)*grid_height
            draw_pts.append([center_x,center_y])
        assert(len(draw_pts)==len(path))
        # print draw_pts

        # XXX: we draw lines to show trajectory
        """
        for ii in xrange(len(draw_pts)-1):
            draw = ImageDraw.Draw(im)
            draw.line([(draw_pts[ii][0],draw_pts[ii][1]),\
                (draw_pts[ii+1][0],draw_pts[ii+1][1])], width=40,fill='red')
            del draw
        """

        # XXX: we draw circles to show trajectory
        for ii in xrange(len(draw_pts)):
            draw = ImageDraw.Draw(im)
            ellipse_radii = 0.25
            draw.ellipse([(draw_pts[ii][0]-ellipse_radii*grid_width,\
                    draw_pts[ii][1]-ellipse_radii*grid_height),\
                    (draw_pts[ii][0]+ellipse_radii*grid_height,\
                    draw_pts[ii][1]+ellipse_radii*grid_height)], fill='black')
            del draw

        # XXX: if there is a point right above the start point,
        # we write 'start' under it
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("./solve_mdp/merriweather/Merriweather UltraBold.ttf", 150)
        draw.text(((starting_pt_col+0.1)*grid_width,\
                (starting_pt_row+0.3)*grid_height), 'start', fill='blue', font=font)
        del draw
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("./solve_mdp/merriweather/Merriweather UltraBold.ttf", 150)
        draw.text(((ending_pt_col-0.4)*grid_width,\
                (ending_pt_row+0.3)*grid_height), 'goal', fill='blue', font=font)
        del draw


        # down sample it
        w = 800
        h = 400
        # http://stackoverflow.com/questions/7936154/python-image-library-clean-downsampling
        if 'P' in im.mode: # check if image is a palette type
            im = im.convert("RGB") # convert it to RGB
            im = im.resize((w,h),Image.ANTIALIAS) # resize it
            im = im.convert("P",dither=Image.NONE, palette=Image.ADAPTIVE) 
            #convert back to palette
        else:
            im = im.resize((w,h),Image.ANTIALIAS) # regular resize
        # im.save(newSourceFile) # save the image to the new source
        # im.save(newSourceFile, quality = 95, dpi=(72,72), optimize = True) 
        # set quality, dpi , and shrink size

        im.save(output_filename+'.png')
        print output_filename+'.png'



