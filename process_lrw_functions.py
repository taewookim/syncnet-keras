from __future__ import print_function

import dlib
import glob
# stackoverflow.com/questions/29718238/how-to-read-mp4-video-to-be-processed-by-scikit-image
import imageio
import math
# import matplotlib
# matplotlib.use('agg')     # Use this for remote terminals, with ssh -X
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
import time
import tqdm

from matplotlib.patches import Rectangle
from skimage.transform import resize

# Facial landmark detection
# http://dlib.net/face_landmark_detection.py.html

from process_lrw_params import *



#############################################################
# EXTRACT AUDIO, FRAMES, AND MOUTHS
#############################################################


# extract_and_save_audio_frames_and_mouths_from_dir



def print_time_till_now(start_time):
    ret = os.system("date")
    till_now = time.time() - start_time
    h = till_now//3600
    m = (till_now - h*3600)//60
    s = (till_now - h*3600 - m*60)//1
    print(h, "hr", m, "min", s, "sec")


def load_detector_and_predictor(verbose=False):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        if verbose:
            print("Detector and Predictor loaded. (load_detector_and_predictor)")
        return detector, predictor
    # If error in SHAPE_PREDICTOR_PATH
    except RuntimeError:
        raise ValueError("\n\nERROR: Wrong Shape Predictor .dat file path - " + \
            SHAPE_PREDICTOR_PATH, "(load_detector_and_predictor)\n\n")


def copy_txt_file(saveDir, videoFileName, verbose=False):
    # Names
    fromFileName = videoFileName
    toFileName = os.path.join(saveDir, "/".join(videoFileName.split("/")[-3:]))
    try:
        shutil.copyfile(fromFileName, toFileName)
        if verbose:
            print("Text file copied:", fromFileName, "->", toFileName,
                "(copy_txt_file)")
        return 0
    except:
        raise ValueError("\n\nERROR: shutil failed to copy " + fromFileName + \
            " to " + toFileName + " (copy_txt_file)\n\n")


def extract_audio_from_mp4(saveDir, videoFileName, dontWriteAudioIfExists, verbose=False):
    # Names
    videoFileName = '.'.join(videoFileName.split('.')[:-1]) + '.mp4'
    audioFileName = os.path.join(saveDir,
        "/".join(videoFileName.split("/")[-3:]).split('.')[0] + ".aac")
    
    # Don't write if .aac file exists
    if dontWriteAudioIfExists:
        # Check if file exists
        if os.path.isfile(audioFileName):
            if verbose:
                print("Audio file, exists, so not written:" + audioFileName + \
                    " (extract_audio_from_mp4)")
            # Return if file exists
            return

    # Just in case, to overwrite or not to overwrite
    if dontWriteAudioIfExists:
        overwriteCommand = '-n'
    else:
        overwriteCommand = '-y'

    # Command
    command = ["ffmpeg", "-loglevel", "error", "-i", videoFileName, "-vn",
               overwriteCommand, "-acodec", "copy", audioFileName]

    # subprocess.call returns 0 on successful run
    try:
        commandReturn = subprocess.call(command)
    except KeyboardInterrupt:
        raise KeyboardInterrupt

    # If audio file could not be written by subprocess
    if commandReturn != 0:
        raise ValueError("\n\nERROR: Audio file " + audioFileName + " NOT WRITEN!! (extract_audio_from_mp4)\n\n")

    if verbose:
        if commandReturn == 0:
            print("Audio file written:", audioFileName, "(extract_audio_from_mp4)")


def extract_and_save_frames_and_mouths(saveDir='lrw',
        videoFileName='/home/voletiv/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.txt',
        extractFramesFromMp4=True,
        writeFrameImages=True,
        detectAndSaveMouths=True,
        dontWriteFrameIfExists=True,
        dontWriteMouthIfExists=True,
        detector=None,
        predictor=None,
        verbose=False):

    # extractFramesFromMp4 and detectAndSaveMouths => Read frames from mp4 video and detect mouths
    # (not extractFramesFromMp4) and detectAndSaveMouths => Read frames from jpeg images and detect mouths
    # extractFramesFromMp4 and (not detectAndSaveMouths) => Read frames from mp4 video
    # (to maybe save them)

    try:
        videoFrames = extract_frames_from_video(videoFileName, verbose)
        

    # If mp4 or jpeg files to read are missing, cascade ValueError up
    except ValueError as err:
        raise ValueError(err)

    # Default face bounding box
    if detectAndSaveMouths:
        # Default face
        face = dlib.rectangle(30, 30, 220, 220)


    # For each frame
    for f, frame in enumerate(videoFrames):

        # If frames are extracted from video, all frames are read
        if detectAndSaveMouths:
            frameNumer = f + 1
        # If frames are read from jpeg images, frame numbers are in their names
        else:
            frameNumer = int(videoFrameNames[f].split('/')[-1].split('.')[0].split('_')[-1])

        # Write the frame image (from video)
        if extractFramesFromMp4 and writeFrameImages:
            write_frame_image(saveDir=saveDir, videoFileName=videoFileName,
                frameNumer=frameNumer, frame=frame,
                dontWriteFrameIfExists=dontWriteFrameIfExists, verbose=verbose)

        # Detect mouths in frames
        if detectAndSaveMouths:
            face = detect_mouth_and_write(saveDir=saveDir,
                videoFileName=videoFileName, frameNumer=frameNumer, frame=frame,
                detector=detector, predictor=predictor,
                dontWriteMouthIfExists=dontWriteMouthIfExists, prevFace=face,
                verbose=verbose)

    return 0


def extract_frames_from_video(videoFileName, verbose=False):
    # Video file name
    videoFileName = '.'.join(videoFileName.split('.')[:-1]) + '.mp4'

    # Handle file not found
    if not os.path.isfile(videoFileName):
        raise ValueError("\n\nERROR: Video file not found:" + videoFileName + \
            "(extract_frames_from_video)\n\n")

    # Read video frames
    videoFrames = imageio.get_reader(videoFileName, 'ffmpeg')

    if verbose:
            print("Frames extracted from video:", videoFileName,
                "(extract_frames_from_video)")

    # Return
    return videoFrames


def read_jpeg_frames_from_dir(saveDir, videoFileName, verbose=False):
    
    # Frame names end with numbers from 00 to 30, so [0-3][0-9]
    videoFrameNamesFormat = os.path.join(saveDir,
                               "/".join(videoFileName.split("/")[-3:]).split('.')[0] + \
                               '_[0-3][0-9].jpg')

    # Read video frame names
    videoFrameNames = sorted(glob.glob(videoFrameNamesFormat))

    try:
        # Read all frame images
        videoFrames = []
        for frameName in videoFrameNames:
            videoFrames.append(imageio.imread(frameName))
    except OSError:
        # If not able to read
        raise ValueError("ERROR: could not read " + frameName + " (read_jpeg_frames_from_dir)")

    if verbose:
            print("Frames read from jpeg images:", videoFileName,
                "(read_jpeg_frames_from_dir)")

    # Return
    return videoFrames, videoFrameNames


def extract_word_frame_numbers(videoFileName, verbose=False):
    # Find the duration of the word_metadata
    wordDuration = extract_word_duration(videoFileName)
    # Find frame numbers
    wordFrameNumbers = range(math.floor(VIDEO_FRAMES_PER_WORD/2 - wordDuration*VIDEO_FPS/2),
        math.ceil(VIDEO_FRAMES_PER_WORD/2 + wordDuration*VIDEO_FPS/2) + 1)
    if verbose:
        print("Word frame numbers = ", wordFrameNumbers, "; Word duration = ", wordDuration)
    return wordFrameNumbers


def extract_word_duration(videoFileName):
    # Read last line of word metadata
    with open(videoFileName) as f:
        for line in f:
            pass
    # Find the duration of the word_metadata`
    return float(line.rstrip().split()[-2])


def write_frame_image(saveDir, videoFileName, frameNumer, frame,
        dontWriteFrameIfExists=True, verbose=False):

    # Name
    frameImageName = os.path.join(saveDir, "/".join(videoFileName.split(
        "/")[-3:]).split('.')[0] + "_{0:02d}".format(frameNumer) + ".jpg")

    # If file is not supposed to be written if it exists
    if dontWriteFrameIfExists:
        # Check if file exists
        if os.path.isfile(frameImageName):
            if verbose:
                print("Frame image exists, so not written:", frameImageName,
                    "(write_frame_image)")
            # Return if file exists
            return

    # Write
    imageio.imwrite(frameImageName, frame)

    if verbose:
        print("Frame image written:", frameImageName, "(write_frame_image)")


def detect_mouth_and_write(saveDir, videoFileName, frameNumer, frame, detector, predictor,
        dontWriteMouthIfExists=True, prevFace=dlib.rectangle(30, 30, 220, 220),
        verbose=False):

    # Image Name
    mouthImageName = os.path.join(saveDir, "/".join(videoFileName.split(
                                  "/")[-3:]).split('.')[0] + \
                                  "_{0:02d}_mouth".format(frameNumer) + ".jpg")

    # If file is not supposed to be written if it exists
    if dontWriteMouthIfExists:
        # Check if file exists
        if os.path.isfile(mouthImageName):
            if verbose:
                print("Mouth image", mouthImageName,
                    "exists, so not detected. (detect_mouth_and_write)")
            # Return if file exists
            return prevFace

    # Detect and save mouth in frame
    mouthImage, face = detect_mouth_in_frame(frame, detector, predictor,
                                             prevFace=prevFace, verbose=verbose)

    # Save mouth image
    imageio.imwrite(mouthImageName, mouthImage)

    if verbose:
        print("Mouth image written:", mouthImageName, "(detect_mouth_and_write)")

    # Return
    return face


def detect_mouth_in_frame(frame, detector, predictor,
                          prevFace=dlib.rectangle(30, 30, 220, 220),
                          verbose=False):
    # Shape Coords: ------> x (cols)
    #               |
    #               |
    #               v
    #               y
    #             (rows)

    # Detect all faces
    faces = detector(frame, 1)

    # If no faces are detected
    if len(faces) == 0:
        if verbose:
            print("No faces detected, using prevFace", prevFace, "(detect_mouth_in_frame)")
        faces = [prevFace]

    # If multiple faces in frame, find the correct face by checking mouth mean
    if len(faces) > 1:

        # Iterate over the faces
        for face in faces:

            # Predict facial landmarks
            shape = predictor(frame, face)

            # # Show landmarks and face
            # win = dlib.image_window()
            # win.set_image(frame)
            # win.add_overlay(shape)
            # win.add_overlay(face)

            # Note all mouth landmark coordinates
            mouthCoords = np.array([[shape.part(i).x, shape.part(i).y]
                                    for i in range(MOUTH_SHAPE_FROM, MOUTH_SHAPE_TO)])

            # Check if correct face is selected by checking position of mouth mean
            mouthMean = np.mean(mouthCoords, axis=0)
            if mouthMean[0] > 110 and mouthMean[0] < 150 \
                    and mouthMean[1] > 140 and mouthMean[1] < 170:
                break

    # If only one face in frame
    else:
        # Note face
        face = faces[0]
        # Predict facial landmarks
        shape = predictor(frame, face)
        # Note all mouth landmark coordinates
        mouthCoords = np.array([[shape.part(i).x, shape.part(i).y]
                                for i in range(MOUTH_SHAPE_FROM, MOUTH_SHAPE_TO)])

    # Mouth Rect: x, y, w, h
    mouthRect = (np.min(mouthCoords[:, 0]), np.min(mouthCoords[:, 1]),
                 np.max(mouthCoords[:, 0]) - np.min(mouthCoords[:, 0]),
                 np.max(mouthCoords[:, 1]) - np.min(mouthCoords[:, 1]))

    # Make mouthRect square
    mouthRect = make_rect_shape_square(mouthRect)

    # Expand mouthRect square
    expandedMouthRect = expand_rect(mouthRect,
        scale=(MOUTH_TO_FACE_RATIO * face.width() / mouthRect[2]),
        frame_shape=(frame.shape[0], frame.shape[1]))

    # Resize to 120x120
    resizedMouthImage = np.round(resize(frame[expandedMouthRect[1]:expandedMouthRect[1] + expandedMouthRect[3],
                                              expandedMouthRect[0]:expandedMouthRect[0] + expandedMouthRect[2]],
                                        (120, 120), preserve_range=True)).astype('uint8')

    # Return mouth
    return resizedMouthImage, face


def make_rect_shape_square(rect):
    # Rect: (x, y, w, h)
    # If width > height
    if rect[2] > rect[3]:
        rect = (rect[0], int(rect[1] + rect[3] / 2 - rect[2] / 2),
                rect[2], rect[2])
    # Else (height > width)
    else:
        rect = (int(rect[0] + rect[2] / 2 - rect[3] / 2), rect[1],
                rect[3], rect[3])
    # Return
    return rect


def expand_rect(rect, scale=1.5, frame_shape=(256, 256)):
    # Rect: (x, y, w, h)
    w = int(rect[2] * scale)
    h = int(rect[3] * scale)
    x = max(0, min(frame_shape[1] - w, rect[0] - int((w - rect[2]) / 2)))
    y = max(0, min(frame_shape[0] - h, rect[1] - int((h - rect[3]) / 2)))
    return (x, y, w, h)


def read_last_line_in_file(videoFileName):
    try:
        with open(videoFileName) as f:
            for line in f:
                 pass
        return line
    except OSError:
        read_last_line_in_file(videoFileName)


