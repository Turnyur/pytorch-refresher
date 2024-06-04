
import os  


data_path = '/home/turnyur/sommer-sem-2024/project/supersizing_3d/code/render/render_output/shapenet_objects/03642806/9c6176af3ee3918d6140d56bf601ecf2'

rendered_files = os.listdir(data_path) 
rendered_files.sort()

file_groups = {'camera': [], 'depth': [], 'render': []}

for file_name in rendered_files:
    file_prefix = file_name.split('_')[0] 
    file_groups[file_prefix].append(file_name)


for j, i in enumerate(file_groups['camera']):
    print(j)
    print()  
    camera_file = file_groups['camera'][i]
    depth_file = file_groups['depth'][i]
    render_file = file_groups['render'][i]

    print(f"C: {camera_file}, D: {depth_file}, R: {render_file}")

    