import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import h5py

class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.index_mapping = {}
        for filename in os.listdir(root_dir):
            if filename.startswith('render'):
                index = int(''.join(char for char in filename.split('_')[-1] if char.isdigit()))
                self.index_mapping[index] = {
                    'render': filename,
                    'depth': f"depth_{index}.png",
                    'camera': f"camera_{index}.mat"
                }
        assert len(self.index_mapping) > 0, "No ShapeNet objects found"

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        
        files = self.index_mapping[idx]
        image_path = os.path.join(self.root_dir, files['render'])
        depth_path = os.path.join(self.root_dir, files['depth'])
        camera_path = os.path.join(self.root_dir, files['camera'])

        # Load shapenet synset
        rgb_image = Image.open(image_path)
        depth = Image.open(depth_path)  
        camera_pose = sio.loadmat(camera_path)
        
        
       
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth = self.transform(depth)
            camera_pose = self.transform(camera_pose['camera_' + str(idx)])
        
        properties = {
            'uuid': '12345', 
            'image': files['render'],
            'depth': files['depth'],
            'camera': 'camera_' + str(idx),
            'id': idx
           
        }

            

        return rgb_image, depth, camera_pose, properties



class ShapeNetDatasetH5(Dataset):
    def __init__(self, h5_file, category,  transform=None):
        self.transform = transform
        self.shapenet_h5 = h5_file
        self.category = category
        self.file_handle = h5py.File(self.shapenet_h5, 'r')

    def __len__(self):

        category_group = self.file_handle[self.category]
        num_images = len([key for key in category_group.keys() if 'render_' in key])
        return num_images

    def __getitem__(self, idx):

        category_group = self.file_handle[self.category]

        # Load camera, depth, and render images
        camera_data = category_group[f'camera_{idx}.mat'][:]
        depth_image = category_group[f'depth_{idx}.png'][()]
        rgb_image = category_group[f'render_{idx}.png'][()]

        if self.transform:
            camera_data = self.transform(camera_data)
            depth_image = self.transform(depth_image)
            rgb_image = self.transform(rgb_image)

        return rgb_image, depth_image, camera_data
        #return  camera_data








test_root_dir = "/home/turnyur/sommer-sem-2024/project/supersizing_3d/code/render/render_output/shapenet_objects/02691156"






#dataset = ShapeNetDataset(test_root_dir, transform=None)
#dataset = ShapeNetDataset(test_root_dir, transform=transforms.ToTensor())
#first_data = dataset[7]
#rgb_images, depths, poses = first_data
#print(rgb_images, depths, poses)
#print(type(rgb_images), type(depths), type(poses))
#print(f'DATASE SIZE: {poses.size()}')

h5_shapnet_file = "./data/shapenet_data.h5"
 #['02691156', '02747177', '02958343', '03642806'] etc
cat = '02691156'
dataset = ShapeNetDatasetH5(h5_shapnet_file, cat, transform=transforms.ToTensor())

BATCH_SIZE = 12
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
shapenetdataiter = iter(dataloader)
images, depths, poses = shapenetdataiter._next_data()
#print(pose)

# for i, (images,_, poses) in enumerate(dataloader):
#     print(poses)
#     print("\n\n")



for i in range(12):
    plt.subplot(3,4, i+1)
    img_np = depths.numpy()[i].transpose((1, 2, 0))
    plt.imshow(img_np, cmap="gray")
    plt.axis('off')
plt.show()



