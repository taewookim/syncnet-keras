import cv2, os, sys, numpy as np
import scipy.io.wavfile as wav
from PIL import Image
import numpy as np
import speechpy
import dlib

from process_lrw_functions import detect_mouth_in_frame, extract_audio_from_mp4
from syncnet_functions import load_pretrained_syncnet_model


def resample_video(source, output, fps=25):
	# https://stackoverflow.com/questions/11004137/re-sampling-h264-video-to-reduce-frame-rate-while-maintaining-high-image-quality
	cmd="ffmpeg -y -loglevel panic -i {} -r {} {}".format(source, fps, output)
	os.system(cmd)
	return output	

# max number of frames in each output
# each output should contain 0.2sec worth of mfcc
EACH_MFCC_OUTPUT_FRAME_SIZE = 20


def extract_mfcc_series(wav_file, target_dir=None):
	(rate, sig) = wav.read(wav_file)

	try:
		mfcc_feat = speechpy.feature.mfcc(sig, sampling_frequency=rate, frame_length=0.010, frame_stride=0.01)
	except IndexError:
		print("index error occurred while extracting mfcc")
		return
	print('sample_rate: {}, mfcc_feat length: {}, mfcc_feat[0] length: {}'.format(rate, len(mfcc_feat), len(mfcc_feat[0])))
	num_output = len(mfcc_feat) / EACH_MFCC_OUTPUT_FRAME_SIZE
	num_output += 1 if (len(mfcc_feat) % EACH_MFCC_OUTPUT_FRAME_SIZE > 0) else 0
	
	# print(mfcc_feat.shape)
	# input(int(num_output))
	images = []

	for index in range(int(num_output)):
		img = Image.new('RGB', (20, 13), "black")
		pixels = img.load()
		for i in range(img.size[0]):
			for j in range(img.size[1]):
				frame_index = index * EACH_MFCC_OUTPUT_FRAME_SIZE + i
				# print(frame_index)
				try:
					if mfcc_feat[frame_index][j] < 0:
						red_amount = min(255, 255 * (mfcc_feat[frame_index][j] / -20))
						pixels[i, j] = (int(red_amount), 0, 0)
					elif (mfcc_feat[frame_index][j] > 0):
						blue_amount = min(255, 255 * (mfcc_feat[frame_index][j] / 20))
						pixels[i, j] = (0, 0, int(blue_amount))
				except IndexError:
					print("index error occurred while extracting mfcc @ " + str(frame_index) + "," + str(j))
					break
		# img.save("{}/mfcc_{:03d}.png".format(target_dir, index), 'PNG')
		
		img_to_np = np.array(img)
		# img_to_np = img_to_np[:,:,0]
		# input(img_to_np.shape)

		images.append(img_to_np)

	return np.asarray(images)


def get_audio_input(video):

	audio_out = "{}.wav".format(video)
	cmd="ffmpeg -y -loglevel panic -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}".format(video, audio_out)
	os.system(cmd)
	
	return extract_mfcc_series(audio_out)

	


def get_video_input(video):

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	cap 		= cv2.VideoCapture(video)
	frameFPS 	= int(cap.get(cv2.CAP_PROP_FPS))
	frameCount 	= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth 	= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# print("FPS: {}".format(frameFPS))
	# print("Frames: {}".format(frameCount))
	# print("Width: {}".format(frameWidth))
	# print("Height: {}".format(frameHeight))
	
	face = dlib.rectangle(30, 30, 220, 220)
	
	lip_model_input = []

	
	while(cap.isOpened()):

		# If frames are extracted from video, all frames are read
		frames = []
		for i in range(5):
			_, frame 	= cap.read()
			if(frame is None):
				break

			mouth, face = detect_mouth_in_frame(
				frame, detector, predictor,
				prevFace=face,
				verbose=False)

			mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY) # convert to grayscale
			mouth = cv2.resize( mouth, (112,112))
			# input(mouth.shape)
			# mouth = mouth[:, :,0] 	# drop the RGB channel
			frames.append(mouth)
			
		# cv2.imshow("Mouth", mouth)
		# if cv2.waitKey(25) & 0xFF == ord('q'):
		# 	cv2.destroyAllWindows()
		# 	break
		
		if(len(frames)==0):
			break

		stacked = np.stack(frames ,axis=2)	#syncnet requires (112,112,5)
		# input(stacked.shape)
		lip_model_input.append(stacked)
	
	return np.asarray(lip_model_input)

if __name__ == '__main__':
		
	
	
	# print(audio_mfcc.shape)
	# print(type(audio_mfcc))
	# print(audio_mfcc.shape)
	# sys.exit(1)
	lip_input = get_video_input("test.mp4")
	# print(type(lip_input))
	# print(lip_input.shape)
	
	audio_input = get_audio_input("test.mp4")
	# print(type(audio_input))
	# print(audio_input.shape)
	# input(">")

	version = 'v4'
	mode = 'both'
	syncnet_audio_model, syncnet_lip_model = load_pretrained_syncnet_model(version=version, mode=mode, verbose=False)

	# print(syncnet_audio_model.summary())
	# input(">")
	# print(syncnet_lip_model.summary())
	# input(">")

	audio_prediction = syncnet_audio_model.predict(audio_input)
	lip_prediction = syncnet_lip_model.predict(lip_input)

	print(audio_prediction)
	input(">")
	print(lip_prediction)