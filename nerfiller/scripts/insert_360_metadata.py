"""
Script to make an equirectangular video compatible with media players, e.g., VLC.
"""

import argparse
import mediapy
import os
from contextlib import ExitStack

from nerfstudio.scripts.render import insert_spherical_metadata_into_file

parser = argparse.ArgumentParser(description="Insert spherical metadata into a video file.")

parser.add_argument("--video_filename", default=None)
parser.add_argument("--image_filename", default=None)

args = parser.parse_args()

if args.video_filename:
    input_filename = args.video_filename
    output_filename = args.video_filename.replace(".mp4", "_360.mp4")

    if os.path.exists(output_filename):
        os.remove(output_filename)

    video = mediapy.read_video(input_filename)
    fps = video.metadata.fps
elif args.image_filename:
    output_filename = os.path.splitext(args.image_filename)[0] + ".mp4"
    image = mediapy.read_image(args.image_filename)
    seconds = 10
    video = [image] * seconds
    fps = 1

with ExitStack() as stack:
    writer = None
    for image in video:
        if writer is None:
            render_width = int(image.shape[1])
            render_height = int(image.shape[0])
            writer = stack.enter_context(
                mediapy.VideoWriter(
                    path=output_filename,
                    shape=(render_height, render_width),
                    fps=fps,
                )
            )
        writer.add_image(image)
insert_spherical_metadata_into_file(output_filename)
