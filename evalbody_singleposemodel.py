import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
from utils import load_graph_model, get_input_tensors, get_output_tensors
import tensorflow as tf
# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


# PATHS
imagePath = 'path/to/.jpg/file'
modelPath = 'path/to/folder/containing/model.json'

# CONSTANTS
OutputStride = 16

KEYPOINT_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]


KEYPOINT_IDS = {name: id for id, name in enumerate(KEYPOINT_NAMES)}

CONNECTED_KEYPOINTS_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_KEYPOINT_INDICES = [(KEYPOINT_IDS[a], KEYPOINT_IDS[b])
                              for a, b in CONNECTED_KEYPOINTS_NAMES]

PART_CHANNELS = [
    'left_face',
    'right_face',
    'left_upper_arm_front',
    'left_upper_arm_back',
    'right_upper_arm_front',
    'right_upper_arm_back',
    'left_lower_arm_front',
    'left_lower_arm_back',
    'right_lower_arm_front',
    'right_lower_arm_back',
    'left_hand',
    'right_hand',
    'torso_front',
    'torso_back',
    'left_upper_leg_front',
    'left_upper_leg_back',
    'right_upper_leg_front',
    'right_upper_leg_back',
    'left_lower_leg_front',
    'left_lower_leg_back',
    'right_lower_leg_front',
    'right_lower_leg_back',
    'left_feet',
    'right_feet'
]


print("Loading model...", end="")
graph = load_graph_model(modelPath)  # downloaded from the link above
print("done.\nLoading sample image...", end="")


def getBoundingBox(keypointPositions, offset=(10, 10, 10, 10)):
    minX = math.inf
    minY = math.inf
    maxX = - math.inf
    maxY = -math.inf
    for x, y in keypointPositions:
        if (x < minX):
            minX = x
        if(y < minY):
            minY = y
        if(x > maxX):
            maxX = x
        if (y > maxY):
            maxY = y
    return (minX - offset[0], minY-offset[1]), (maxX+offset[2], maxY + offset[3])


# load sample image into numpy array
img = tf.keras.preprocessing.image.load_img(imagePath)
imgWidth, imgHeight = img.size

targetWidth = (int(imgWidth) // OutputStride) * OutputStride + 1
targetHeight = (int(imgHeight) // OutputStride) * OutputStride + 1

print(imgHeight, imgWidth, targetHeight, targetWidth)
img = img.resize((targetWidth, targetHeight))
x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
InputImageShape = x.shape
print("Input Image Shape in hwc", InputImageShape)


widthResolution = int((InputImageShape[1] - 1) / OutputStride) + 1
heightResolution = int((InputImageShape[0] - 1) / OutputStride) + 1
print('Resolution', widthResolution, heightResolution)

# Get input and output tensors
input_tensor_names = get_input_tensors(graph)
print(input_tensor_names)
output_tensor_names = get_output_tensors(graph)
print(output_tensor_names)
input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

# Preprocessing Image
# For Resnet
if any('resnet_v1' in name for name in output_tensor_names):
    # add imagenet mean - extracted from body-pix source
    m = np.array([-123.15, -115.90, -103.06])
    x = np.add(x, m)
# For Mobilenet
elif any('MobilenetV1' in name for name in output_tensor_names):
    x = (x/127.5)-1
else:
    print('Unknown Model')
sample_image = x[tf.newaxis, ...]
print("done.\nRunning inference...", end="")

# evaluate the loaded model directly
with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(output_tensor_names, feed_dict={
                       input_tensor: sample_image})
print("done. {} outputs received".format(len(results)))  # should be 8 outputs

for idx, name in enumerate(output_tensor_names):
    if 'displacement_bwd' in name:
        print('displacement_bwd', results[idx].shape)
    elif 'displacement_fwd' in name:
        print('displacement_fwd', results[idx].shape)
    elif 'float_heatmaps' in name:
        heatmaps = np.squeeze(results[idx], 0)
        print('heatmaps', heatmaps.shape)
    elif 'float_long_offsets' in name:
        longoffsets = np.squeeze(results[idx], 0)
        print('longoffsets', longoffsets.shape)
    elif 'float_short_offsets' in name:
        offsets = np.squeeze(results[idx], 0)
        print('offests', offsets.shape)
    elif 'float_part_heatmaps' in name:
        partHeatmaps = np.squeeze(results[idx], 0)
        print('partHeatmaps', partHeatmaps.shape)
    elif 'float_segments' in name:
        segments = np.squeeze(results[idx], 0)
        print('segments', segments.shape)
    elif 'float_part_offsets' in name:
        partOffsets = np.squeeze(results[idx], 0)
        print('partOffsets', partOffsets.shape)
    else:
        print('Unknown Output Tensor', name, idx)


# BODYPART SEGMENTATION
partOffsetVector = []
partHeatmapPositions = []
partPositions = []
partScores = []
partMasks = []

# Segmentation MASk
segmentation_threshold = 0.7
segmentScores = tf.sigmoid(segments)
mask = tf.math.greater(segmentScores, tf.constant(segmentation_threshold))
print('maskshape', mask.shape)
segmentationMask = tf.dtypes.cast(mask, tf.int32)
segmentationMask = np.reshape(
    segmentationMask, (segmentationMask.shape[0], segmentationMask.shape[1]))
print('maskValue', segmentationMask[:][:])

plt.clf()
plt.title('Segmentation Mask')
plt.ylabel('y')
plt.xlabel('x')
plt.imshow(segmentationMask * OutputStride)
plt.show()

# Draw Segmented Output
mask_img = Image.fromarray(segmentationMask * 255)
mask_img = mask_img.resize(
    (targetWidth, targetHeight), Image.LANCZOS).convert("RGB")
mask_img = tf.keras.preprocessing.image.img_to_array(
    mask_img, dtype=np.uint8)

segmentationMask_inv = np.bitwise_not(mask_img)
fg = np.bitwise_and(np.array(img), np.array(
    mask_img))
plt.title('Foreground Segmentation')
plt.imshow(fg)
plt.show()
bg = np.bitwise_and(np.array(img), np.array(
    segmentationMask_inv))
plt.title('Background Segmentation')
plt.imshow(bg)
plt.show()


# Part Heatmaps, PartOffsets,
for i in range(partHeatmaps.shape[2]):

    heatmap = partHeatmaps[:, :, i]  # First Heat map
    heatmap[np.logical_not(tf.math.reduce_any(mask, axis=-1).numpy())] = -1
    # Set portions of heatmap where person is not present in segmentation mask, set value to -1

    # SHOW HEATMAPS

    plt.clf()
    plt.title('Heatmap: ' + PART_CHANNELS[i])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.imshow(heatmap * OutputStride)
    plt.show()

    heatmap_sigmoid = tf.sigmoid(heatmap)
    y_heat, x_heat = np.unravel_index(
        np.argmax(heatmap_sigmoid, axis=None), heatmap_sigmoid.shape)

    partHeatmapPositions.append([x_heat, y_heat])
    partScores.append(heatmap_sigmoid[y_heat, x_heat].numpy())
    # Offset Corresponding to heatmap x and y
    x_offset = partOffsets[y_heat, x_heat, i]
    y_offset = partOffsets[y_heat, x_heat, partHeatmaps.shape[2]+i]
    partOffsetVector.append([x_offset, y_offset])

    key_x = x_heat * OutputStride + x_offset
    key_y = y_heat * OutputStride + y_offset
    partPositions.append([key_x, key_y])


print('partheatmapPositions', np.asarray(partHeatmapPositions).shape)
print('partoffsetVector', np.asarray(partOffsetVector).shape)
print('partkeypointPositions', np.asarray(partPositions).shape)
print('partkeyScores', np.asarray(partScores).shape)


# POSE ESTIMATION
offsetVector = []
heatmapPositions = []
keypointPositions = []
keyScores = []
for i in range(heatmaps.shape[2]):
    heatmap = heatmaps[:, :, i]  # First Heat map
    # SHOW HEATMAPS
    '''
    plt.clf()
    plt.title('Heatmap' + str(i) + KEYPOINT_NAMES[i])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.imshow(heatmap * OutputStride)
    plt.show()
    '''

    heatmap_sigmoid = tf.sigmoid(heatmap)
    y_heat, x_heat = np.unravel_index(
        np.argmax(heatmap_sigmoid, axis=None), heatmap_sigmoid.shape)

    heatmapPositions.append([x_heat, y_heat])
    keyScores.append(heatmap_sigmoid[y_heat, x_heat].numpy())
    # Offset Corresponding to heatmap x and y
    x_offset = offsets[y_heat, x_heat, i]
    y_offset = offsets[y_heat, x_heat, heatmaps.shape[2]+i]

    offsetVector.append([x_offset, y_offset])
    key_x = x_heat * OutputStride + x_offset
    key_y = y_heat * OutputStride + y_offset
    keypointPositions.append([key_x, key_y])


print('heatmapPositions', np.asarray(heatmapPositions).shape)
print('offsetVector', np.asarray(offsetVector).shape)
print('keypointPositions', np.asarray(keypointPositions).shape)
print('keyScores', np.asarray(keyScores).shape)

# PRINT KEYPOINT CONFIDENCE SCORES
print("Keypoint Confidence Score")
for i, score in enumerate(keyScores):
    print(KEYPOINT_NAMES[i], score)

# PRINT POSE CONFIDENCE SCORE
print("Pose Confidence Score", np.mean(np.asarray(keyScores)))

# Get Bounding BOX
(xmin, ymin), (xmax, ymax) = getBoundingBox(
    keypointPositions, offset=(0, 0, 0, 0))
print("Bounding Box xmin, ymin, xmax, ymax format: ", xmin, ymin, xmax, ymax)

# Show Bounding BOX
implot = plt.imshow(img)
# Get the current reference
ax = plt.gca()
# Create a Rectangle patch
rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                         linewidth=1, edgecolor='r', facecolor='none', fill=False)

# Add the patch
ax.add_patch(rect)
plt.show()

# Show all keypoints

plt.figure(0)
im = plt.imread(imagePath)
implot = plt.imshow(im)

x_points = []
y_points = []
for i, [x, y] in enumerate(keypointPositions):
    x_points.append(x)
    y_points.append(y)
plt.scatter(x=x_points, y=y_points, c='r', s=40)
plt.show()


# DEBUG KEYPOINTS
#  Show Each Keypoint and it's name
'''
for i, [x, y] in enumerate(keypointPositions):
    plt.figure(i)
    plt.title('keypoint' + str(i) + KEYPOINT_NAMES[i])
#    img = plt.imread(imagePath)
    implot = plt.imshow(img)

    plt.scatter(x=[x], y=[y], c='r', s=40)
    plt.show()
'''

# SHOW CONNECTED KEYPOINTS
plt.figure(20)
for pt1, pt2 in CONNECTED_KEYPOINT_INDICES:
    plt.title('connection points')
    implot = plt.imshow(img)
    plt.plot((keypointPositions[pt1][0], keypointPositions[pt2][0]), (
        keypointPositions[pt1][1], keypointPositions[pt2][1]), 'ro-', linewidth=2, markersize=5)
plt.show()
