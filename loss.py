import torch
import numpy as np
from metrics import AlignmentMetrics
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

    @staticmethod
    def patch2patch_kernel_alignment_score(feats_A, feats_B):
        """
        Compute the patch2patch kernel alignment score between two sets of features.
        Use a copy from the metrics.py to retain the gradient computation.
        feats_A: B, N, D
        feats_B: B, N, E # can be different from dimension
        """
        # normalize the features along the last dimension
        feats_A = F.normalize(feats_A, dim=-1)
        feats_B = F.normalize(feats_B, dim=-1)

        # compute the kernel matrix --> patch2patch similarity matrix for both A and B # B, N, N
        kernel_matrix_A = feats_A @ feats_A.transpose(1, 2)
        kernel_matrix_B = feats_B @ feats_B.transpose(1, 2)

        # normalize the rows for both kernel matrices
        kernel_matrix_A = F.normalize(kernel_matrix_A, dim=-1)
        kernel_matrix_B = F.normalize(kernel_matrix_B, dim=-1)

        # compute the similarity of the kernel matrices between A and B
        # Since each row is now a unit vector, the dot product of corresponding rows
        # will be 1 if they are identical.
        alignment_score = (kernel_matrix_A * kernel_matrix_B).sum(dim=-1)  # B, N

        # average the alignment score across the patches (dim=1) and then across the samples (dim=0)
        alignment_score = alignment_score.mean(dim=1).mean(dim=0)
        return alignment_score

    @staticmethod
    def sample2sample_kernel_alignment_score(feats_A, feats_B):
        """
        Compute the sample2sample kernel alignment score between two sets of features.
        Use a copy from the metrics.py to retain the gradient computation.
        feats_A: B, N, D
        feats_B: B, N, E
        """
        # take the mean across last dimension # B, D
        feats_A = feats_A.mean(dim=-2)
        feats_B = feats_B.mean(dim=-2)

        # normalize the features along the last dimension
        feats_A = F.normalize(feats_A, dim=-1)
        feats_B = F.normalize(feats_B, dim=-1)

        # compute the kernel matrix --> sample2sample similarity matrix for both A and B # B, B
        kernel_matrix_A = feats_A @ feats_A.transpose(0, 1)
        kernel_matrix_B = feats_B @ feats_B.transpose(0, 1)

        # normalize the rows for both kernel matrices
        kernel_matrix_A = F.normalize(kernel_matrix_A, dim=-1)
        kernel_matrix_B = F.normalize(kernel_matrix_B, dim=-1)

        # compute the similarity of the kernel matrices between A and B
        # Since each row is now a unit vector, the dot product of corresponding rows
        # will be 1 if they are identical.
        alignment_score = (kernel_matrix_A * kernel_matrix_B).sum(dim=-1)  # B

        # average the alignment score across the samples
        alignment_score = alignment_score.mean(dim=0)
        return alignment_score

    def __call__(self, model, images, model_kwargs=None, zs=None, alignment_kwargs=None):
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
        model_output, zs_tilde, fs_tilde, all_layer_feats = model(
            model_input, time_input.flatten(), use_projection=True,
            return_all_layers=alignment_kwargs["compute_alignment"] and alignment_kwargs["max_score_across_layers"],
            **model_kwargs
        )
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection and kernel alignment loss
        proj_loss, kernel_alignment_loss = 0., 0.

        bsz = zs[0].shape[0]
        # REPA loss
        proj_loss = 0.
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        # Kernel Alignment across patches
        if self.loss_type == "patch2patch":
            # NOTE: We should compute kernel alignment with unprojected features only
            for i, (z, f_tilde) in enumerate(zip(zs, fs_tilde)):
                # NOTE: The loss should be the negative of the alignment score (minimize the negative alignment score)
                kernel_alignment_loss += -self.patch2patch_kernel_alignment_score(z, f_tilde)
            kernel_alignment_loss /= len(zs)

        elif self.loss_type == "sample2sample":
            # NOTE We should compute kernel alignment with unprojected features only
            for i, (z, f_tilde) in enumerate(zip(zs, fs_tilde)):
                # NOTE: The loss should be the negative of the alignment score (minimize the negative alignment score)
                kernel_alignment_loss += -self.sample2sample_kernel_alignment_score(z, f_tilde)
            kernel_alignment_loss /= len(zs)

        elif self.loss_type is not None:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        alignment_scores = None

        if alignment_kwargs["compute_alignment"]:
            alignment_scores = {}
            with torch.no_grad():
                # Iterate through all metrics
                for metric in alignment_kwargs["log_alignment_metrics"]:

                    # If we want to get the max score across all layers
                    if alignment_kwargs["max_score_across_layers"]:
                        curr_metric_scores = []
                        assert all_layer_feats is not None, "All layer features should be computed for logging all layers"
                        # Iterate through layers
                        for curr_layer_feat in all_layer_feats:
                            curr_alignment_score = 0.
                            for z, f_tilde in zip(zs, curr_layer_feat):
                                measure_kwargs = {}
                                if "kernel_alignment" in metric:
                                    # Our method assumes [B, L, D]
                                    feats_A = z
                                    feats_B = f_tilde
                                else:
                                    # Rep paper method assumes [B, D]
                                    feats_A = z.mean(dim=1)
                                    feats_B = f_tilde.mean(dim=1)
                                
                                if metric == "cknna":
                                    # cknna needs topk
                                    measure_kwargs["topk"] = alignment_kwargs["cknna_topk"]
                                elif metric == "patch2patch_kernel_alignment_score_jsd":
                                    # jsd needs temperature
                                    measure_kwargs["temperature"] = alignment_kwargs["p2p_jsd_temp"]
                                elif metric == "sample2sample_kernel_alignment_score_jsd":
                                    # jsd needs temperature
                                    measure_kwargs["temperature"] = alignment_kwargs["s2s_jsd_temp"]

                                curr_score = AlignmentMetrics.measure(
                                    metric=metric,
                                    feats_A=feats_A,
                                    feats_B=feats_B,
                                    **measure_kwargs
                                )
                                curr_alignment_score += curr_score

                            curr_alignment_score /= len(zs)
                            curr_metric_scores.append(curr_alignment_score)

                        # Get the max metric score across layers and move it to GPU for reduce
                        max_metric_score = max(curr_metric_scores)
                        max_metric_score = torch.tensor(max_metric_score, device=z.device)
                        alignment_scores[metric] = max_metric_score

                    # Otherwise, we just compute the alignment score for the aligned layer (fs_tilde)
                    else:
                        curr_alignment_score = 0.
                        for z, f_tilde in zip(zs, fs_tilde):
                            # NOTE: For the CKNNA score, we take the mean across patches, to get a single feature vector for each sample
                            measure_kwargs = {}
                            if "kernel_alignment" in metric:
                                # Our method assumes [B, L, D]
                                feats_A = z
                                feats_B = f_tilde
                            else:
                                # Rep paper method assumes [B, D]
                                feats_A = z.mean(dim=1)
                                feats_B = f_tilde.mean(dim=1)

                            if metric == "cknna":
                                # cknna needs topk
                                measure_kwargs["topk"] = alignment_kwargs["cknna_topk"]
                            elif metric == "patch2patch_kernel_alignment_score_jsd":
                                # jsd needs temperature
                                measure_kwargs["temperature"] = alignment_kwargs["p2p_jsd_temp"]
                            elif metric == "sample2sample_kernel_alignment_score_jsd":
                                # jsd needs temperature
                                measure_kwargs["temperature"] = alignment_kwargs["s2s_jsd_temp"]

                            curr_score = AlignmentMetrics.measure(
                                metric=metric,
                                feats_A=feats_A,
                                feats_B=feats_B,
                                **measure_kwargs
                            )
                            curr_alignment_score += curr_score

                        # Convert it to tensor for reduce
                        curr_alignment_score /= len(zs)
                        curr_alignment_score = torch.tensor(curr_alignment_score, device=z.device)
                        alignment_scores[metric] = curr_alignment_score

        return denoising_loss, proj_loss, kernel_alignment_loss, alignment_scores
