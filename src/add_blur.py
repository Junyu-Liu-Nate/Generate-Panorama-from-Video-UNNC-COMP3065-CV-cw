import cv2
import numpy as np

# def motion_kernel(d, sz=65):
#     """
#     Generate a horizontal motion blur kernel.

#     :param d: The length of the motion blur kernel.
#     :param sz: The size of the motion blur kernel.
#     :return: The motion blur kernel.
#     """
#     kern = np.zeros((sz, sz))
#     kern[sz//2, sz//2-d//2:sz//2+d//2] = 1
#     kern /= d
#     return kern

def motion_kernel(d, angle, sz=65):
    # Create a horizontal motion blur kernel
    kern = np.zeros((sz, sz))
    kern[sz//2, sz//2-d//2:sz//2+d//2] = 1
    kern /= d
    
    # Rotate the kernel by the specified angle
    center = (sz // 2, sz // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    kern = cv2.warpAffine(kern, rot_matrix, (sz, sz))
    
    return kern

print('There are 4 test videos choices: test_video_1, test_video_2, test_video_3, test_video_4')
test_name = input('Please type which video you want to input: ')
while test_name not in ['test_video_1', 'test_video_2', 'test_video_3', 'test_video_4']:
    test_name = input('Invalid input. Please type which video you want to input: ')
print(f'You have chosen {test_name} to add blur.')
orignal_name = test_name

print('There are 3 motion blur magnitude for your chosen video name: 10, 20, 30')
magnitude = input('Please type which magnitude: ')
while magnitude not in ['10', '20', '30']:
    magnitude = input('Invalid input. Please type which magnitude: ')
test_name = test_name + '_blur_' + magnitude

angle = input('Please type the direction of the blur (in range [0,360)): ')
while int(angle) not in range(0,360):
    angle = input('Invalid input. Please type the direction of the blur (in range [0,360)): ')
test_name = test_name + '_' + angle
print(f'The blurred video will be {test_name}.')
print('')  

# test_name = 'test_video_1'
# d = 30
# ang = 0
d = int(magnitude)
ang = int(angle)
in_vid_path = '../vid/' + orignal_name + '.mp4'
out_vid_path = '../vid/' + test_name + '.mp4'

# Open the input video
cap = cv2.VideoCapture(in_vid_path)

# Get the video frame size and frame rate
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_vid_path, fourcc, frame_rate, (frame_width, frame_height))

# Generate a horizontal motion blur kernel
# psf = motion_kernel(d)
psf = motion_kernel(d, ang)

# Process the video frame by frame
while cap.isOpened():
    # Read a frame from the input video
    ret, frame = cap.read()
    if not ret:
        break

    # Split the frame into its color channels
    b, g, r = cv2.split(frame)

    # Apply motion blur to each color channel
    b_blurred = cv2.filter2D(b, -1, psf)
    g_blurred = cv2.filter2D(g, -1, psf)
    r_blurred = cv2.filter2D(r, -1, psf)

    # Merge the blurred color channels back into a single frame
    frame_blurred = cv2.merge([b_blurred, g_blurred, r_blurred])

    # Write the blurred frame to the output video
    out.write(frame_blurred)

# Release the input and output video objects
cap.release()
out.release()
