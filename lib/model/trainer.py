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

from .ray import exponential_integration
from ..utils.metrics import psnr

# Warning: you MUST NOT change the resolution of marching cube
RES = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(nn.Module):

    def __init__(self, config, model, pe_coord, pe_view, log_dir):
        super().__init__()

        self.cfg = config
        self.pos_encoder_coord = pe_coord.to(device)
        self.pos_encoder_view = pe_view.to(device)
        self.model = model.to(device)
        self.N_importance = config.num_pts_importance_per_ray
        self.N_importance_render = config.num_pts_importance_per_ray_render
        self.log_dir = log_dir
        self.log_dict = {}

        self.init_optimizer_scheduler()
        self.rgb_loss_func = nn.MSELoss()
        self.init_log_dict()

    def init_optimizer_scheduler(self):
        """Define Optimizer"""
        trainable_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.cfg.lr, 
            betas=(self.cfg.beta1, self.cfg.beta2), weight_decay=self.cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 200*5, gamma=0.5)

    def init_log_dict(self):
        """Custom log dict. """
        self.log_dict['total_loss'] = 0.0
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['image_count'] = 0



    def sample_pdf_points(self, bins, weights, N_samples, det=False):
        """sample points from cdf given cummutive weight
        Args:
            bins: segmentations along a ray [B, Nr, Np-1]
            weights: weigths along a ray [B, Nr, Np-2]
        """
        weights = weights.squeeze(-1) + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True) # [B, Nr, Np-2, 1]
        cdf = torch.cumsum(pdf, -1) # [B, Nr, Np-2]
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], dim=-1)  # [B, Nr, Np-1]

        # Take uniform samples is det == True
        if det:
            u = torch.linspace(0., 1., steps=N_samples)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
        u = u.to(cdf.device).contiguous() # [B, Nr, N_sample]

        # Invert CDF - get indices along CDF where values in u would be placed
        inds = torch.searchsorted(cdf, u, right=True)
        # Clamp out of bounds indices
        below = torch.clamp(inds-1, min=0) # = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.clamp(inds, max=cdf.shape[-1]-1) # = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # [B, Nr, N_samples, 2]

        # Sample from cdf and the corresponding bin centers.
        matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]] # [B, Nr, N_samples, Np-1]
        # cdf_g, bins_g: [B, Nr, N_samples, 2]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)
        bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)

        # Convert samples to ray length
        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples


    def sample_evenly(self, ray_orig, ray_dir, near=1.0, far=3.0, num_points=64):
        """Sample points along rays.
        Args:
            ray_orig (torch.FloatTensor): Origin of the rays [B, Nr, 3].
            ray_dir (torch.FloatTensor): Direction of the rays [B, Nr, 3].
            near, far (float): Near and Far plane of the camera.
            num_points (int): Number of points (Np) to sample along the rays.
         Returns:
            points (torch.FloatTensor): 3D coordinates of the points [B, Nr, Np, 3].
            z_vals (torch.FloatTensor): Depth values of the points [B, Nr, Np, 1].
            deltas (torch.FloatTensor): Distance between the points [B, Nr, Np, 1].
        """
        B, Nr = ray_orig.shape[:2]

        # split into small segments and sample in each segment
        t = torch.linspace(0.0, 1.0, num_points, device=ray_orig.device).view(1, 1, -1) + \
            (torch.rand([B, Nr, num_points], device=ray_orig.device)/ num_points) # [B, Nr, Np]

        z_vals = near * (1.-t) + far * t # sampled depth between near and far [B, Nr, Np]

        points = ray_orig[..., None, :] + ray_dir[..., None, :] * z_vals[..., None]
        deltas = z_vals.diff(dim=-1, prepend=(torch.zeros(B, Nr, 1, device=z_vals.device)+ near))
        # deltas = deltas * torch.norm(ray_dir[..., None, :], dim=-1)

        return points, z_vals[..., None], deltas[..., None]
    
    def sample_hierarchical(self, ray_orig, ray_dir, z_vals, weights, near=1.0, far=3.0, num_importance=64):
        """Draw samples from PDF using z_vals as bins and weights as probabilities.
        """
        B, Nr = ray_orig.shape[:2]
        
        z_vals = z_vals.squeeze(-1)
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        new_samples = self.sample_pdf_points(z_vals_mid, weights[..., 1:-1], num_importance, det=False)

        # Resample points from ray based on PDF 
        z_vals_hiera, _ = torch.sort(torch.cat([z_vals, new_samples.detach()], dim=-1), dim=-1) # [B, Nr, N_importance + Np]
        points = ray_orig[..., None, :] + ray_dir[..., None, :] * z_vals_hiera[..., None]  # [B, N_rays, Np + NHp, 3]
        deltas = z_vals_hiera.diff(dim=-1, prepend=(torch.zeros(B, Nr, 1, device=z_vals.device)+ near))
        # deltas = deltas * torch.norm(ray_dir[..., None, :], dim=-1)

        return points, z_vals_hiera[..., None], deltas[..., None]

    def predict_radience(self, coords, ray_dir=None):
        """Predict radiance at the given coordinates.
        Args:
            coords (torch.FloatTensor): 3D coordinates of the points [B, Nr, Np, 3].
            ray_dir (torch.FloatTensor): directions of the rays [B, Nr, 3].
        Returns:
            rgb (torch.FloatTensor): Radiance at the given coordinates of shape [B, Nr, Np, 3].
            sigma (torch.FloatTensor): volume density at the given coordinates of shape [B, Nr, Np, 1].
        """
        if len(coords.shape) == 2:
            coords_pe = self.pos_encoder_coord(coords)
        else:
            coords_pe = self.pos_encoder_coord(coords.view(-1, 3)).view(*coords.shape[:-1], -1)

        if ray_dir is None and len(coords.shape) == 2:
            ray_dir = torch.tensor([0.0, 0.0, 1.0]).view(1, 3).expand(coords.shape[0], 3).to(device)
        else: 
            # fill ray_dir for all coords in the same ray
            ray_dir = ray_dir[..., None, :].repeat(1, 1, coords.shape[-2], 1)
        
        # normalize the view dir before embedding
        # ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)

        if len(ray_dir.shape) == 2:
            ray_dir_pe = self.pos_encoder_view(ray_dir)
        else:
            ray_dir_pe = self.pos_encoder_view(ray_dir.view(-1, 3)).view(*ray_dir.shape[:-1], -1)

        # ray_dir_pe - torch.Size([2, 512, 128, 33])
        rgb, sigma = self.model(coords_pe, ray_dir_pe)
        return rgb, sigma

    def volume_render(self, rgb, sigma, depth, deltas):
        """Ray marching to compute the radiance at the given rays.
        Args:
            rgb (torch.FloatTensor): Radiance at the sampled points of shape [Batch_size, Num_ray, Num_point, 3].
            sigma (torch.FloatTensor): Volume density at the sampled points of shape [B, Nr, Np, 1].
            deltas (torch.FloatTensor): Distance between the points of shape [B, Nr, Np, 1].
        Returns:
            ray_colors (torch.FloatTensor): Radiance at the given rays of shape [B, Nr, 3].
            weights (torch.FloatTensor): Weights of the given rays of shape [B, Nr, 1].
        """
        # TODO: Try out different neural rendering methods.
        # density * length of segment
        tau = sigma * deltas
        ray_colors, ray_depth, weight_sum, weigths = exponential_integration(rgb, tau, depth, exclusive=True)
        return ray_colors, ray_depth, weight_sum, weigths 


    def forward(self):
        """Forward pass of the network. 
        Returns: rgb (torch.FloatTensor): Ray codors of shape [Batch_size, Num_ray, 3].
        """
        B, Nr = self.ray_orig.shape[:2]

        # Step 1 : Sample points along the rays
        self.coords, self.z_vals, self.deltas = self.sample_evenly(
            self.ray_orig, self.ray_dir, self.cfg.near, self.cfg.far, self.cfg.num_pts_per_ray
        )
        # Step 2 : Predict radiance and volume density at the sampled points
        self.rgb, self.sigma = self.predict_radience(self.coords, self.ray_dir)
        # Step 3 : Volume rendering to compute the RGB color at the given rays
        self.ray_colors, self.ray_depth, self.ray_weight_sum, self.ray_weigths = self.volume_render(
            self.rgb, self.sigma, self.z_vals, self.deltas
        )

        # if use hierarchical sampling
        if self.N_importance > 0:
            # Sample additional points along the rays hierarchically
            self.coords, self.z_vals, self.deltas = self.sample_hierarchical(
                self.ray_orig, self.ray_dir, self.z_vals, self.ray_weigths, self.cfg.near, self.cfg.far, self.N_importance
            )
            # Predict radiance and volume density
            self.rgb, self.sigma = self.predict_radience(self.coords, self.ray_dir)
            # Volume rendering
            self.ray_colors, self.ray_depth, self.ray_weight_sum, _ = self.volume_render(
                self.rgb, self.sigma, self.z_vals, self.deltas
            )

        # Step 4 : Compositing with background color
        if self.cfg.bg_color == 'white':
            bg = torch.ones(B, Nr, 3, device=self.ray_colors.device)
            self.rgb = (1.0 - self.ray_weight_sum) * bg + self.ray_weight_sum * self.ray_colors
        else:
            self.rgb = self.ray_weight_sum * self.ray_colors

    def backward(self):
        """Backward pass of the network. """
        loss = 0.0
        # rgb_loss = torch.abs(self.rgb - self.img_gts).mean()
        rgb_loss = self.rgb_loss_func(self.rgb, self.img_gts)
        # TODO: You can also desgin your own loss function.
        loss = rgb_loss # + any other loss terms

        self.log_dict['rgb_loss'] += rgb_loss.item()
        self.log_dict['total_loss'] += loss.item()
        loss.backward()

    def step(self, data):
        """Training step. """
        # Get rays, and put them on the device
        self.ray_orig = data['rays'][..., :3].to(device)
        self.ray_dir = data['rays'][..., 3:].to(device)
        self.img_gts = data['imgs'].to(device)

        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()
        # self.scheduler.step()

        self.log_dict['total_iter_count'] += 1
        self.log_dict['image_count'] += self.ray_orig.shape[0]

    def log(self, step, epoch):
        """Log the training information.
        """
        log_text = 'STEP {}-EPOCH {}/{}'.format(step, epoch, self.cfg.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | total loss: {:>.6E}'.format(self.log_dict['total_loss'])
        self.log_dict['rgb_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | rgb loss: {:>.6E}'.format(self.log_dict['rgb_loss'])

        log.info(log_text)

        for key, value in self.log_dict.items():
            if 'loss' in key:
                wandb.log({key: value}, step=step)
        self.init_log_dict()






    def render(self, ray_orig, ray_dir):
        """Render a 2D image for evaluation.
        """
        B, Nr = ray_orig.shape[:2]
        # sample points for each ray
        coords, z_vals, deltas = self.sample_evenly(ray_orig, ray_dir, self.cfg.near, self.cfg.far, )
        # get radience for each sample point in each ray 
        rgb, sigma = self.predict_radience(coords, ray_dir)
        # render for each ray
        ray_colors, ray_depth, weight_sum, weights = self.volume_render(rgb, sigma, z_vals, deltas)

        # Sample additional points along the rays hierarchically
        if self.N_importance_render > 0:
            coords, z_vals, deltas = self.sample_hierarchical(
                ray_orig, ray_dir, z_vals, weights, self.cfg.near, self.cfg.far, self.N_importance_render
            )
            rgb, sigma = self.predict_radience(coords, ray_dir)
            ray_colors, ray_depth, weight_sum, _ = self.volume_render(rgb, sigma, z_vals, deltas)
        
        if self.cfg.bg_color == 'white':
            bg = torch.ones(B, Nr, 3, device=ray_colors.device)
            render_img = (1.0 - weight_sum) * bg + weight_sum * ray_colors
        else:
            render_img = weight_sum * ray_colors

        return render_img, ray_depth, weight_sum

    def reconstruct_3D(self, save_dir, epoch=0, sigma_threshold = 50., chunk_size=8192):
        """Reconstruct the 3D shape from the volume density. """

        # Mesh evaluation
        window_x = torch.linspace(-1., 1., steps=RES, device='cuda')
        window_y = torch.linspace(-1., 1., steps=RES, device='cuda')
        window_z = torch.linspace(-1., 1., steps=RES, device='cuda')
        
        coord = torch.stack(torch.meshgrid(window_x, window_y, window_z)).permute(1, 2, 3, 0).reshape(-1, 3).contiguous()

        _points = torch.split(coord, int(chunk_size), dim=0)
        voxels = []
        for _p in _points:
            _, sigma = self.predict_radience(_p, ray_dir=None) 
            voxels.append(sigma)
        voxels = torch.cat(voxels, dim=0)

        np_sigma = torch.clip(voxels, 0.0).reshape(RES, RES, RES).cpu().numpy()

        vertices, faces = mcubes.marching_cubes(np_sigma, sigma_threshold)
        #vertices = ((vertices - 0.5) / (res/2)) - 1.0
        vertices = (vertices / (RES-1)) * 2.0 - 1.0

        h = trimesh.Trimesh(vertices=vertices, faces=faces)
        h.export(os.path.join(save_dir, '%04d.obj' % (epoch)))

    def validate(self, loader, img_shape, step=0, epoch=0, sigma_threshold = 50., chunk_size=8192, save_img=False):
        """validation function for generating final results. """
        
        torch.cuda.empty_cache() # To avoid CUDA out of memory
        self.eval()

        log.info(f"Start validation using {len(loader)} images at resolution {img_shape[0]}x{img_shape[1]}")

        self.valid_mesh_dir = os.path.join(self.log_dir, "mesh")
        if not os.path.exists(self.valid_mesh_dir):
            os.makedirs(self.valid_mesh_dir)
        if save_img:
            self.valid_img_dir = os.path.join(self.log_dir, "img")
            if not os.path.exists(self.valid_img_dir):
                os.makedirs(self.valid_img_dir)

        psnr_total = 0.0
        wandb_img, wandb_img_gt = [], []

        with torch.no_grad():
            # Evaluate 3D reconstruction
            self.reconstruct_3D(self.valid_mesh_dir, epoch=epoch,
                sigma_threshold=sigma_threshold, chunk_size=chunk_size
            )
            log.info(f"Reconstruction result saved to {self.valid_mesh_dir}")

            # Evaluate 2D novel view rendering
            for i, data in enumerate(tqdm(loader)):
                rays = data['rays'].to(device)          # [1, Nr, 6]
                img_gt = data['imgs'].to(device)        # [1, Nr, 3]
                mask = data['masks'].repeat(1, 1, 3).to(device)

                _rays = torch.split(rays, int(chunk_size), dim=1)
                pixels = []
                for _r in _rays:
                    ray_orig = _r[..., :3]          # [1, chunk, 3]
                    ray_dir = _r[..., 3:]           # [1, chunk, 3]
                    ray_rgb, ray_depth, ray_alpha = self.render(ray_orig, ray_dir)
                    pixels.append(ray_rgb)
                pixels = torch.cat(pixels, dim=1) # [1, chunk, 3]
                psnr_total += psnr(pixels, img_gt)

                img = (pixels).reshape(*img_shape, 3).cpu().numpy() * 255
                wandb_img.append(wandb.Image(img))

                # gt = (img_gt).reshape(*img_shape, 3).cpu().numpy() * 255
                # wandb_img_gt.append(wandb.Image(gt))
                if save_img:
                    # Image.fromarray(gt.astype(np.uint8)).save(
                    #     os.path.join(self.valid_img_dir, "gt-{:04d}-{:03d}.png".format(epoch, i)) )
                    Image.fromarray(img.astype(np.uint8)).save(
                        os.path.join(self.valid_img_dir, "img-{:04d}-{:03d}.png".format(epoch, i)) )
            if save_img:
                log.info(f"Rendering images saved to {self.valid_img_dir}")

        wandb.log({"Rendered Images": wandb_img}, step=step)
        # wandb.log({"Ground-truth Images": wandb_img_gt}, step=step)
                
        psnr_total /= len(loader)

        log_text = 'EPOCH {}/{}'.format(epoch, self.cfg.epochs)
        log_text += ' {} | {:.2f}'.format(f"PSNR", psnr_total)

        wandb.log({'PSNR': psnr_total}, step=step)
        log.info(log_text)
        self.train()

    def save_model(self, epoch):
        """Save the model checkpoint. """
        fname = os.path.join(self.log_dir, f'model-{epoch}.pth')
        log.info(f'Saving model checkpoint to: {fname}')
        torch.save(self.model, fname)

    