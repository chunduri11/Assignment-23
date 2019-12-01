import sys
from PIL import Image

for i in range(308):
    images = [Image.open(x) for x in ['orig_rotated_frames/frame' + str(i) + '.jpg', '68point_landmarks/frame' + str(i) + '.jpg', 'stabilized/frame' + str(i) +'.jpg']]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save('resized_combined/frame'+ str(i) + '.jpg')