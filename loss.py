import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            loss_type="cos_sim",
        ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.loss_type = loss_type

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            # randomly sample the timestep for reverse diffusion loss
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        # timestep for the reverse diffusion loss
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        # random initial noise
        noises = torch.randn_like(images)

        # forward diffusion
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        # Noise prediction, features after projection, features before projection
        model_output, zs_tilde, fs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection and kernel alignment loss
        proj_loss, ka_loss = 0., 0.

        bsz = zs[0].shape[0]
        if "cos_sim" in self.loss_type:
            # Avoid the average is taken for previously computed losses, use another variable to store the current loss
            curr_proj_loss = 0.
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    curr_proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            curr_proj_loss /= (len(zs) * bsz)
            proj_loss += curr_proj_loss

        # Kernel Alignment across patches
        if "ka_patch" in self.loss_type:
            # Avoid the average is taken for previously computed losses, use another variable to store the current loss
            curr_ka_loss = 0.
            for i, (z, f_tilde) in enumerate(zip(zs, fs_tilde)):
                # Compute the semantic relation activation matrices A -> (B x L x L), normalize the representation row-wise with L2 norm
                # The shape of the kernel matrix should be [L x L] for each data in the batch.
                a_mat = F.normalize(z @ z.transpose(1, 2), dim=-1)
                a_tilde_mat = F.normalize(f_tilde @ f_tilde.transpose(1, 2), dim=-1)
                # Compute the element-wise loss
                curr_ka_loss += torch.sqrt(F.mse_loss(a_mat, a_tilde_mat, reduction='mean'))
            curr_ka_loss /= len(zs)
            ka_loss += curr_ka_loss

        if "ka_sample" in self.loss_type:
            raise NotImplementedError("The sample-wise projection loss is not implemented yet.")

        if "ka_channel" in self.loss_type:
            raise NotImplementedError("The channel-wise projection loss is not implemented yet.")

        return denoising_loss, proj_loss, ka_loss
