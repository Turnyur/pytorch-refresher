import hydra
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import os
import os.path as osp
import torch
import h5py

cs = ConfigStore.instance()

@dataclass
class DataConfig:
    bs_train: int = 1
    bs_val: int = 1
    num_workers: int = 0

@dataclass
class Shapenet3DData(DataConfig):
    train_split: float = 0.95
    validation_split: float = 0.05
    encoder_dir: str = (
        "/learning_vision3d/datasets/renders"
    )
    rgb_dir: str = "/learning_vision3d/datasets/renders"
    class_ids: str = "02858304,02924116,03790512,04468005,\
    02992529,02843684,02954340,02691156,\
    02933112,03001627,03636649,04090263,\
    04379243,04530566,02828884,02958343,\
    03211117,03691459,04256520,04401088,\
    02747177,02773838,02801938,02808440,\
    02818832,02834778,02871439,02876657,\
    02880940,02942699,02946921,03085013,\
    03046257,03207941,03261776,03325088,\
    03337140,03467517,03513137,03593526,\
    03624134,03642806,03710193,03759954,\
    03761084,03797390,03928116,03938244,\
    03948459,03991062,04004475,04074963,\
    04099429,04225987,04330267,04460130,04554684"
    name: str = "warehouse3d"
    bs_train: int = 8
    bs_val: int = 1
    bs_test: int = 1
    num_workers: int = 0

cs.store(name="default", node=DataConfig)
cs.store(name="shapenet", node=Shapenet3DData)

@dataclass
class LoggingConfig:
    log_dir: str = "job_outputs"
    name: str = "temp"

cs.store(name="logging", node=LoggingConfig)

@dataclass
class ModelConfig:
    encoder: str = ""
    decoder: str = ""
    c_dim: int = 0
    inp_dim: int = 3
    fine_tune: str = "all"  # "encoder" or "decoder" or "none"

cs.store(name="model", node=ModelConfig)

@dataclass
class ResourceConfig:
    gpus: int = 1
    num_nodes: int = 1
    num_workers: int = 0
    accelerator: str = "ddp"  # ddp or dp or none

    # cluster specific config
    use_cluster: bool = False
    max_mem: bool = True  # if true use volta32gb for SLURM jobs.
    time: int = 60 * 36  # minutes
    partition: str = "dev"
    comment: str = "please fill this if using priority partition"

    mesh_th: float = 2.0

cs.store(name="resource", node=ResourceConfig)


@dataclass
class RenderConfig:
    img_size: int = 128
    focal_length: int = 300
    near_plane: float = 0.1
    far_plane: float = 2.5
    camera_near_dist: float = 1.3
    camera_far_dist: float = 1.7
    cam_num: int = 5  # if -1, render on fly, dont use prerend
    num_pre_rend_masks: int = 50  # -1 corresponds to use all
    ray_num_per_cam: int = 340
    on_ray_num_samples: int = 80
    rgb: bool = True
    normals: bool = False
    depth: bool = False

    # No camera pose params
    softmin_temp: float = 1.0
    loss_mode: str = "softmax"  # other option is "softmax"
    use_momentum: bool = True


cs.store(name="render", node=RenderConfig)




@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    shapenet: Shapenet3DData = field(default_factory=Shapenet3DData)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

cs.store(name="config", node=Config)




@hydra.main(config_name="config", version_base=None)
def synth_pretraining(cfg: Config):
    
    #print(OmegaConf.to_yaml(cfg.data))
    #print(cfg.logging.name)

    _curr_path = osp.dirname(osp.abspath(__file__))
    _base_path = _curr_path  # osp.join(_curr_path, "..")

    #log_dir = osp.join(_base_path, cfg.logging.log_dir, cfg.logging.name)
    #os.makedirs(log_dir, exist_ok=True)
    #OmegaConf.save(cfg, osp.join(log_dir, "config.txt"))

    synsets = cfg.shapenet.class_ids.split(",")
    synsets.sort()

    #[f for f in os.listdir(osp.join(_base_path)) if len(f) > 3]

    #synsetModels = [s for s in synsets]
    file_handle = h5py.File("./data/shapenet_data.h5", 'r')
    synsets = list(file_handle.keys()) #data_cfg.class_ids.split(",") 
    synsets.sort()


    synsetModels = [
            [f for f in file_handle[s] ] for s in synsets
        ]

    paths = []
    for i in range(len(synsets)):
        for m in synsetModels[i]:
            paths.append([synsets[i], m])
    # paths end

    #print(paths)

    train_size = int(cfg.shapenet.train_split * len(paths))
    validation_size = int(cfg.shapenet.validation_split * len(paths))
    test_size = len(paths) - train_size - validation_size
    (
        train_split,
        validation_split,
        test_split,
    ) = torch.utils.data.random_split(
        paths, [train_size, validation_size, test_size]
    )
    # print(
    #     "Total Number of paths:",
    #     len(paths),
    #     len(train_split),
    #     len(validation_split),
    # )
    print(f"Dataset: {len(paths)}")
    print(f"Train split: {len(train_split)}")
    print(f"Validation split: {len(validation_split)}")
    print(f"Test split: {len(test_split)}")

    #print(synsetModels)
    index = 15
    #print(type(train_split))
    #rel_path = os.path.join(train_split[index][0], train_split[index][1])
    print(os.path.join(train_split[index][0], train_split[index][1]))
    #print(train_split[index])
    depth_0 = '/home/turnyur/sommer-sem-2024/project/supersizing_3d/code/render/render_output/shapenet_objects/02691156/depth_0.png'
    

    


if __name__ == "__main__":
    synth_pretraining()


