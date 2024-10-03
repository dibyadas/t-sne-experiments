from PIL import Image
import glob

# File paths for the PNG images
# png_files = sorted(glob.glob("pics/*.png"), key=lambda x: int(x[5:-4])) 
png_files = glob.glob("pics_torch_opt_perfect/*.png")
png_files = sorted(glob.glob("pics_torch_opt_perfect/*.png"), key=lambda x: int(x[23:-4])) 
# Open images and store them in a list
images = [Image.open(file) for file in png_files]

# Save as a GIF
images[0].save('torch_opt_perfect.gif',
               save_all=True,
               append_images=images[1:], 
               duration=100,    # Duration between frames in milliseconds
               loop=0)          # Loop count, 0 for infinite loop
