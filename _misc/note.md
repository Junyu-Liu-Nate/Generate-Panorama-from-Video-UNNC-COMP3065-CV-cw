# Note

## Edge region of the panaroma is blurred

When stitching images into a panorama using feature-based methods such as SIFT, it is possible for the resulting panorama to have blurred regions near the edges. This can happen when the images being stitched together were not taken from the same viewpoint or when the camera was not rotated around its nodal point. In these cases, the homography calculated by the stitching algorithm may not be able to perfectly align the images, resulting in blurred regions near the edges of the panorama.

One way to reduce this type of blur is to use a different projection method when stitching the images together. For example, instead of projecting all images onto a single image plane, you can use a cylindrical or spherical projection to better preserve the relative geometry of the images.

## Code Not working

One thing to note is that your code currently calculates the homography between each pair of adjacent frames independently. This can result in accumulated errors when stitching many frames together. To reduce these errors, you can calculate the cumulative homography from the first frame to each subsequent frame and use that to warp the frames.

If the code is not stitching the frames from your video into a panorama well, there could be several reasons for this. Here are some potential issues and suggestions for how to improve the stitching:

- **Poor feature detection and matching**: The quality of the panorama depends on the accuracy of the feature detection and matching between adjacent frames. If the SIFT features are not being detected or matched accurately, the calculated homographies may be incorrect, resulting in a poor-quality panorama. To improve the feature detection and matching, you can try using a different feature detector (such as ORB or SURF), adjusting the parameters of the feature detector and matcher, or using a different matching strategy (such as a ratio test to filter out low-quality matches).
- **Incorrect focal length**: The focal length specified in the code must be accurate for the cylindrical projection to work correctly. If the focal length is incorrect, the warped frames may not align properly, resulting in a poor-quality panorama. To improve the accuracy of the focal length, you can try calibrating your camera to determine its exact focal length in pixels.
- **Non-planar scene**: The homography-based stitching method used in the code assumes that the scene being captured is planar (i.e., all points lie on a single plane). If this assumption is not met (e.g., if the scene contains significant depth variation), the calculated homographies may be incorrect, resulting in a poor-quality panorama. To improve the stitching in this case, you can try using a different stitching method that can handle non-planar scenes (such as bundle adjustment).
  