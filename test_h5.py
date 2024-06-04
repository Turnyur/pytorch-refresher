import h5py
import numpy as np

# Example data
data_entries = [
    {
        'rgb_image': np.random.rand(256, 256, 3),
        'depth_image': np.random.rand(256, 256),
        'camera_pose': np.random.rand(4, 4),
        'properties': {
            'uuid': '12345',
            'image_filename': 'render_0.png',
            'depth_filename': 'depth_0.png',
            'camera_filename': 'camera_0.mat',
            'id': '1'
        }
    },
    {
        'rgb_image': np.random.rand(256, 256, 3),
        'depth_image': np.random.rand(256, 256),
        'camera_pose': np.random.rand(4, 4),
        'properties': {
            'uuid': '67890',
            'image_filename': 'render_1.png',
            'depth_filename': 'depth_1.png',
            'camera_filename': 'camera_1.mat',
            'id': '2'
        }
    }

]

# Create an HDF5 file and write the data
with h5py.File("./data/shapenet_data.h5", "w") as f:
    for idx, data in enumerate(data_entries):
        entry_group = f.create_group(f'entry_{idx}')
        
        # Save RGB image
        entry_group.create_dataset('rgb_image', data=data['rgb_image'])
        
        # Save depth image
        entry_group.create_dataset('depth_image', data=data['depth_image'])
        
        # Save camera pose
        entry_group.create_dataset('camera_pose', data=data['camera_pose'])
        
        # Save properties as attributes
        properties_group = entry_group.create_group('properties')
        properties_group.attrs['uuid'] = data['properties']['uuid']
        properties_group.attrs['image_filename'] = data['properties']['image_filename']
        properties_group.attrs['depth_filename'] = data['properties']['depth_filename']
        properties_group.attrs['camera_filename'] = data['properties']['camera_filename']
        properties_group.attrs['id'] = data['properties']['id']



# with h5py.File("./data/shapenet_data.h5", "r") as f:
#     for entry_key in f.keys():
#         entry_group = f[entry_key]
        
#         rgb_image = entry_group['rgb_image'][()]
#         depth_image = entry_group['depth_image'][()]
#         camera_pose = entry_group['camera_pose'][()]
        
#         properties = entry_group['properties'].attrs
#         properties_dict = {key: properties[key] for key in properties.keys()}
        
#         print(f"Entry: {entry_key}")
#         print(f"RGB Image Shape: {rgb_image.shape}")
#         print(f"Depth Image Shape: {depth_image.shape}")
#         print(f"Camera Pose: {camera_pose}")
#         print(f"Properties: {properties_dict}")
