import numpy as np
import cv2

# Sample matrix
camera_matrix = {
    'K': np.array([[1.7321, 0, 0, 0],
                   [0, 1.7321, 0, 0],
                   [0, 0, -1.0000, -0.1000],
                   [0, 0, -1.0000, 0]]),
    'RT': np.array([[0.9688, 0.1863, 0.1632, 0.1632],
                    [0, 0.6590, -0.7522, -0.7522],
                    [-0.2477, 0.7287, 0.6384, 0.6384],
                    [0, 0, 0, 1.0000]])
}

# Extract intrinsic matrix K
K = camera_matrix['K']
focal_length = K[0, 0] 

# Extract extrinsic matrix RT
RT = camera_matrix['RT']
rotation_matrix = RT[:3, :3]
translation_vector = RT[:3, 3]


elevation_angle = np.arcsin(-rotation_matrix[1, 2])
azimuth_angle = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])


distance = np.linalg.norm(translation_vector)


near_plane = distance - 0.1
far_plane = distance + 0.1

# Print the extracted parameters
print("Focal Length:", focal_length)
print("Elevation Angle:", np.degrees(elevation_angle))
print("Azimuth Angle:", np.degrees(azimuth_angle))
print("Distance:", distance)
print("Near Plane:", near_plane)
print("Far Plane:", far_plane)