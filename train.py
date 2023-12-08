import os
import logging as log
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import trimesh
import mcubes
import wandb

from torch.utils.data import DataLoader
from lib.datasets.multiview_dataset import MultiviewDataset
from lib.model.trainer import Trainer

from lib.model.positional_encoding import PositionalEncoding as PE
from lib.model.nerf import NERF
from lib.model.idr import ImplicitDifferentiableRenderer as IDR

from lib.utils.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_archive(save_dir, config):

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(config, os.path.join(tmpdir, 'config.yaml'))
        shutil.copy('train.py', os.path.join(tmpdir, 'train.py'))
        shutil.copy('test.py', os.path.join(tmpdir, 'test.py'))
        shutil.copytree(
            os.path.join('lib'), os.path.join(tmpdir, 'lib'),
            ignore=shutil.ignore_patterns('__pycache__')
        )
        shutil.make_archive(
            os.path.join(save_dir, 'code_copy'), 'zip', tmpdir
        ) 


def main(config):
    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # setup log dir
    log_dir = config.log_dir

    # Backup code.
    create_archive(log_dir, config.config)

    # Setup wandb
    if config.wandb_id is not None:
        wandb_id = config.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(log_dir, 'wandb_id.txt'), 'w+') as f:
            f.write(wandb_id)

    wandb_mode = "disabled" if (not config.wandb) else "online"
    wandb.init(
        id=wandb_id, project=config.wandb_name, config=config,
        name=os.path.basename(log_dir), resume="allow",
        settings=wandb.Settings(start_method="fork"),
        mode=wandb_mode, dir=log_dir,
        tags=[os.path.basename(config.data_root), config.exp_name]
    )
    

    # Initialize dataset and dataloader.
    dataset = MultiviewDataset(config.data_root, mip=0, bg_color=config.bg_color,
                                sample_rays=True, n_rays=config.num_rays_per_img)
    valid_dataset = MultiviewDataset(config.data_root, mip=2, split='val')

    loader = DataLoader(dataset=dataset, batch_size=config.batch_size, 
                        shuffle=True, num_workers=config.workers, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1,
                        shuffle=False, num_workers=config.workers, pin_memory=True)
    
    rander_shape = valid_dataset.img_shape # Height and width of the rendered image

    # ================= Initialize network and trainer =================
    pe_coord = PE(config.num_freq_coord, config.max_freq_coord)
    pe_view = PE(config.num_freq_view, config.max_freq_view)

    # network_mlp = MLP(pe.out_dim, config.out_dim, activation=get_activation_class(config.activation), 
    #               num_layers= config.num_layers , hidden_dim=config.hidden_dim,
    #               skip=[config.skip] )
    network_nerf = NERF(input_dim=pe_coord.out_dim, view_input_dim=pe_view.out_dim, 
        num_layers=config.num_layers, hidden_dim=config.hidden_dim, actv1=config.actv1,
        skip1=config.skip1, feature_dim=config.feature_dim, 
        num_layers_view=config.num_layers_view, hidden_dim_view=config.hidden_dim_view,
        actv2=config.actv2, output_actv=config.out_actv
    )

    # network_nerf = IDR(input_dim=pe_coord.out_dim, input_dim2=pe_view.out_dim, 
    #     num_layers=config.num_layers, hidden_dim=config.hidden_dim,
    #     actv1=config.actv1, skip1=config.skip1, feature_dim=config.feature_dim, 
    #     num_layers2=config.num_layers_view, hidden_dim2=config.hidden_dim_view,
    #     actv2=config.actv2, skip2=config.skip1, output_actv=config.out_actv
    # )

    # pick a network
    network = network_nerf
    wandb.watch(network)
    trainer = Trainer(config, network, pe_coord, pe_view, log_dir)
    if config.num_pts_importance_per_ray > 0:
        logging.info(f'Use hierarchical sampling - n_importance: {config.num_pts_importance_per_ray}, n_importance_render: {config.num_pts_importance_per_ray_render}.')

    # ======================= Main training loop =======================
    global_step = 0
    for epoch in range(config.epochs):
        # training
        for data in loader:
            trainer.step(data)
            if global_step % config.log_every == 0:
                trainer.log(global_step, epoch)
            global_step += 1

        # validate  or epoch == 0 
        if (epoch+1) % config.valid_every == 0: 
            trainer.validate(valid_loader, rander_shape, step=global_step, epoch=epoch, 
                sigma_threshold=config.sigma_thres, chunk_size=config.chunk_size, save_img=config.save_img)
        
        # save model
        if (epoch+1) % config.save_every == 0:
            trainer.save_model(epoch)

    wandb.finish()


if __name__ == "__main__":
    parser = parse_options()
    args, args_str = argparse_to_str(parser)

    args.log_dir = os.path.join(args.save_root, args.exp_name,
        f'{datetime.now().strftime("%Y%m%d-%H%M")}')

    # log_path = os.path.join(args.log_dir, "logs.txt")
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format='%(asctime)s|%(levelname)8s| %(message)s',
        level=args.log_level, handlers=handlers
    )
    logging.info(f'Config: \n{args_str}')
    main(args)