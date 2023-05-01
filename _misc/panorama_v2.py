import cv2

# Example usage
test_name = 'video_1'

vid_path = 'vid/' + test_name + '.mp4'
img_path = 'img/' + test_name + '.jpg'

# Load the video
cap = cv2.VideoCapture(vid_path)

# Create a list to store the frames
frames = []

# Read the frames from the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Release the video capture
cap.release()

# Create a Stitcher object
stitcher = cv2.Stitcher_create()

# Stitch the frames together
status, pano = stitcher.stitch(frames)

# Check if the stitching was successful
if status == cv2.Stitcher_OK:
    # # Show the panorama
    # cv2.imshow('Panorama', pano)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('Panorama.jpg', pano)
else:
    print('Error stitching the frames')
