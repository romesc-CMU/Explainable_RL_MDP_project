# http://stackoverflow.com/questions/5324647/how-to-merge-a-transparent-png-image-with-another-image-using-pil
from PIL import Image, ImageOps, ImageDraw

images = ['./tiles/bad_road.png', './tiles/forest.png', './tiles/grass.png', \
    './tiles/ground.png', './tiles/house1.png', './tiles/house2.png', \
    './tiles/house3.png', './tiles/house4.png', './tiles/human.png', \
    './tiles/ice.png', './tiles/road.png', './tiles/rock.png', \
    './tiles/sand.png', './tiles/traffic.png', './tiles/water.png']

images = ['./tiles/bad_road.png', './tiles/grass.png', './tiles/ice.png', \
    './tiles/road.png', './tiles/rock.png', './tiles/sand.png', \
    './tiles/traffic.png']

for i in images:
    im = Image.open(i)
    # crop it into a circle
    # http://stackoverflow.com/questions/890051/how-do-i-generate-circular-thumbnails-with-pil
    size = (120, 120)
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    output = ImageOps.fit(im, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)
    # paste it onto a background image
    background = Image.open('./tiles/ground.png')
    background.paste(output, (0, 0), output)
    # background.show()
    background.save(i[:-4]+'_circle.png')

# images=['./tiles/human2.png','./tiles/tree.png']
# for i in images:
#     im = Image.open(i)
#     # crop it into a circle
#     # http://stackoverflow.com/questions/890051/how-do-i-generate-circular-thumbnails-with-pil
#     size = (120, 120)
#     # mask = Image.new('L', size, 0)
#     # draw = ImageDraw.Draw(mask)
#     # draw.ellipse((0, 0) + size, fill=255)
#     # output = ImageOps.fit(im, mask.size, centering=(0.5, 0.5))
#     # output.putalpha(mask)
    
#     # paste it onto a background image
#     background = Image.open('./tiles/grass.png')
#     background.paste(im, (0, 0), im)
#     # background.show()
#     background.save(i[:-4]+'_on_grass.png')
