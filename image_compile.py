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

# Parameters -----------------------------------------
source_dir = '/path/to/images' # Source directory to the parent folder to all timepoint folders
out_dir = '/path/to/output' # Output folder for all avi files
datetime_filename = 'datetime.csv' # Dates and times of acquisitions for each replicates/timepoints
fps = 30
resolution = (1024, 1026)
timepoint_len = 600
codec = 'FFV1' 
cores = 4
# ----------------------------------------------------

# Retrieve paths for compilation
timepoint_folders = natsorted(glob.glob(source_dir + '/*'), alg=ns.IGNORECASE)

unique_ids = {}
for t in timepoint_folders:
    for p in natsorted(glob.glob(t + '/*'), alg=ns.IGNORECASE):
        id = re.sub(t + '/', '', p)

        if id == 'display_and_comments.txt':
            continue

        if id not in unique_ids.keys():
            unique_ids[id] = []

        unique_ids[id].append(p)

# Retrieve date time data
summary = dict(
    replicate=[],
    time=[],
    timepoints=[])

frame_level = dict(
    replicate=[],
    frame=[],
    time=[],
    timepoints=[])

for replicate, folders in unique_ids.items():
    dt = []
    timepoints = []
    for i,f in enumerate(folders):
        if 'metadata.txt' not in glob.glob(f'{f}/*'):
            summary['time'].append(None)
            summary['replicate'].append(replicate)
            summary['timepoints'].append(i)
            continue

        with open(f'{f}/metadata.txt', 'r+') as d:
            metadata = json.load(d)

            summary['time'].append(metadata['Summary']['Time'][:-6])
            summary['replicate'].append(replicate)
            summary['timepoints'].append(i)

            for j,frame_t in enumerate([frame for frame in metadata.keys() if 'Frame' in frame]):
                frame_level['replicate'].append(replicate)
                frame_level['frame'].append(j)
                frame_level['time'].append(metadata[frame_t]['Time'][:-6])
                frame_level['timepoints'].append(i)

datetime_df = pd.DataFrame(data=data)
datetime_df.to_csv(f'{out_dir}/{datetime_filename}')

# Frame producers and consumers
def consume(chunk, replicates_pg, frame_queue, frame_queue_id, request_queue):
    for replicate, folders in chunk:
        writer = vuba.Writer(f'{out_dir}/{replicate}{replicate_extension}.avi', resolution=resolution, fps=fps, codec=codec, grayscale=False)

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

                    replicates_pg[replicate].update(1)

                    if release:
                        break

        writer.close()

    request_queue.put('Finished.')

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

            video = vuba.Video(f'{folder}/*.tif')

            # For acquisitions where failure occurred during capture
            if len(video) < timepoint_len:
                print(folder)
                frame_queues[frame_queue_id].put('Skip.')
            else:
                frames = np.empty((timepoint_len, resolution[1], resolution[0]), dtype='uint8')
                for f,frame in tqdm(enumerate(video.read(0, len(video), grayscale=True)), desc='Reader', total=len(video), leave=False):
                    frames[f,:,:] = frame.astype('uint8')

                skip = False
                for frame in frames:
                    if frame is None:
                        skip = True

                if skip:
                    frame_queues[frame_queue_id].put('Skip.')
                else:
                    frame_queues[frame_queue_id].put(frames)

                del frames
                gc.collect()


chunks = divide(cores, unique_ids.items())

replicates = list(unique_ids.keys())
replicates_pg = {}
for i, r in enumerate(replicates): 
    replicates_pg[r] = tqdm(total=len(unique_ids[r]), desc=r, position=i+1)

tqdm.set_lock(mp.RLock())

request_queue = mp.Queue()
frame_queues = {}
for i, chunk in enumerate(chunks):
    frame_queue_id = f'P{i}'
    frame_queue = mp.Queue()
    proc = mp.Process(target=consume, args=(chunk, replicates_pg, frame_queue, frame_queue_id, request_queue))
    proc.start()

    frame_queues[frame_queue_id] = frame_queue

produce(frame_queues, request_queue)
