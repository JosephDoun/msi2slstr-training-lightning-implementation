import sys
import imageio
import os
import re
    
from glob import glob


def main(args):
    # Find image files in provided folder.
    files = [file for file in glob(os.path.join(args[0], "*.*"))
             # Support jpeg or png formats.
             if re.match(".*.(jpe?g|png)", file)]

    # Data container.
    gif_content = []
    
    # Load arrays.
    for image in sorted(files, key=lambda x: x.split(".")[0]):
        gif_content.append(imageio.v3.imread(image))

    # Build gif.
    imageio.mimsave(os.path.join(args[0], "demo_preview.gif"), gif_content,
                    duration=len(gif_content) * 200,
                    loop=65535)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
