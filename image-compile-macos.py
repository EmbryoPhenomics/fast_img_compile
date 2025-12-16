# Script for iterating over multiple replicates and compiling them each into individual avi files
# i.e. one avi per replicate will be produced by the script, for the timepoints specified.
# Note this script is for [timepoint -> position] folder structure
# e.g.
# - timepoint_1
#   - A1
#   - A2
# etc.

import vuba
from tqdm import tqdm
import glob
import os
from natsort import natsorted, ns
import re
import multiprocessing as mp
from more_itertools import divide
import time
import gc
import numpy as np
import pandas as pd
import datetime
import json

# Parameters -----------------------------------------
source_dir = "/Users/otills/Downloads/Oli_Compile"  # Source directory to the parent folder to all timepoint folders
out_dir = "/Users/otills/Desktop/"  # Output folder for all avi files
summary_datetime_filename = "summary_datetime.csv"  # Dates and times of acquisitions for each replicates/timepoints (filename only)
frame_datetime_filename = "frame_datetime.csv"  # Dates and times of individual frames for each replicates/timepoints (filename only)
replicate_extension = "" # E.g. _pt1, _pt2 etc. if video for the same experiment is split across multiple drives
fps = 30 # Frame-rate source video was captured at 
resolution = (1024, 1024) # Source video resolution
timepoint_len = 300 # Number of frames captured for each individual, per timepoint
codec = "FFV1"  # Lossless FFMPEG encoder
cores = 6 # Number of cores to use for parallel processing
# ----------------------------------------------------

# Retrieve paths for compilation
timepoint_folders = natsorted(glob.glob(source_dir + "/*"), alg=ns.IGNORECASE)

unique_ids = {}
for t in timepoint_folders:
    for p in natsorted(glob.glob(t + "/*"), alg=ns.IGNORECASE):
        id = re.sub(t + "/", "", p)

        if id == "display_and_comments.txt":
            continue

        if id not in unique_ids.keys():
            unique_ids[id] = []

        unique_ids[id].append(p)

print("[INFO] Checking for existing complete compiled videos...")
print("Before checking: " + str(len(unique_ids.keys())))

video_lens = {}
for k in list(unique_ids.keys()):
    source_timepoint_len = len(unique_ids[k])

    out_file = vuba.Video(f"{out_dir}/{k}{replicate_extension}.avi")
    out_len = len(out_file)
    out_file.close()
    out_timepoint_len = round(out_len / 300)

    if source_timepoint_len == out_timepoint_len:
        del unique_ids[k]

print("After checking: " + str(len(unique_ids.keys())))

# Retrieve date time data
summary = dict(replicate=[], time=[], timepoints=[])

frame_level = dict(
    replicate=[], frame=[], elapsed_time_s=[], elapsed_time_ms=[], timepoints=[]
)
print(str(frame_level))
for replicate, folders in unique_ids.items():
    print(str(folders))
    dt = []
    timepoints = []
    for i, f in enumerate(folders):
        if not os.path.exists(f"{f}/metadata.txt"):
            summary["time"].append(None)
            summary["replicate"].append(replicate)
            summary["timepoints"].append(i)
            continue

        with open(f"{f}/metadata.txt", "r") as d:
            metadata = json.load(d)

            summary["time"].append(metadata["Summary"]["Time"][:-6])
            summary["replicate"].append(replicate)
            summary["timepoints"].append(i)

            current_time = 0
            for j, frame_t in enumerate(
                [frame for frame in metadata.keys() if "Frame" in frame]
            ):
                if j == 0:
                    current_time = metadata[frame_t]["ElapsedTime-ms"]

                relative_time = metadata[frame_t]["ElapsedTime-ms"] - current_time
                frame_level["replicate"].append(replicate)
                frame_level["frame"].append(j)
                frame_level["elapsed_time_s"].append(round(relative_time / 1000, 2))
                frame_level["elapsed_time_ms"].append(round(relative_time, 2))
                frame_level["timepoints"].append(i)

summary_datetime_df = pd.DataFrame(data=summary)
summary_datetime_df.to_csv(f"{out_dir}/{summary_datetime_filename}")

frame_datetime_df = pd.DataFrame(data=frame_level)
frame_datetime_df.to_csv(f"{out_dir}/{frame_datetime_filename}")

# Frame producers and consumers
def consume(chunk, frame_queue, frame_queue_id, request_queue):
    for replicate, folders in chunk:
        writer = vuba.Writer(
            f"{out_dir}/{replicate}{replicate_extension}.avi",
            resolution=resolution,
            fps=fps,
            codec=codec,
            grayscale=False,
        )

        for f in folders:
            request_queue.put((frame_queue_id, f))

            while True:
                frames = frame_queue.get()

                if type(frames) == str:
                    break
                else:
                    if frames is not None:
                        release = True

                    for frame in frames:
                        writer.write(vuba.bgr(frame))

                    del frames
                    gc.collect()

                   # replicates_pg[replicate].update(1)

                    if release:
                        break

        writer.close()

    request_queue.put("Finished.")


def produce(frame_queues, request_queue):
    finished_counter = 0

    while True:

        if finished_counter >= cores:
            break

        out = request_queue.get()

        if type(out) == str:
            finished_counter += 1

        else:
            frame_queue_id, folder = out

            video = vuba.Video(f"{folder}/*.tif")

            # For acquisitions where failure occurred during capture
            if len(video) < timepoint_len:
                print(folder)
                frame_queues[frame_queue_id].put("Skip.")
            else:
                frames = np.empty(
                    (timepoint_len, resolution[1], resolution[0]), dtype="uint8"
                )
                for f, frame in tqdm(
                    enumerate(video.read(0, len(video), grayscale=True)),
                    desc="Reader",
                    total=len(video),
                    leave=False,
                ):
                    frames[f, :, :] = frame.astype("uint8")

                skip = False
                for frame in frames:
                    if frame is None:
                        skip = True

                if skip:
                    frame_queues[frame_queue_id].put("Skip.")
                else:
                    frame_queues[frame_queue_id].put(frames)

                del frames
                gc.collect()


if __name__ == "__main__":
    chunks = list(divide(cores, list(unique_ids.items())))

    request_queue = mp.Queue()
    frame_queues = {}

    for i, chunk in enumerate(chunks):
        frame_queue_id = f"P{i}"
        frame_queue = mp.Queue()

        proc = mp.Process(
            target=consume,
            args=(chunk, frame_queue, frame_queue_id, request_queue),
        )
        proc.start()

        frame_queues[frame_queue_id] = frame_queue

    produce(frame_queues, request_queue)
