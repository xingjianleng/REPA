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
    def patch2patch_kernel_alignment_score(feats_A, feats_B, feats_A_=None, feats_B_=None, detach_grad=False):
        """
        Compute the patch2patch kernel alignment score between two sets of features.
        Use a copy from the metrics.py to retain the gradient computation.
        feats_A: B, N, D
        feats_B: B, N, E # can be different from dimension
        feats_A_: (B x n_gpus, D) (if provided)
        feats_B_: (B x n_gpus, E) (if provided)
        detach_grad: bool, detach the gradient of the part of the feature matrix
        """
        # normalize the features along the last dimension
        feats_A = F.normalize(feats_A, dim=-1)
        feats_B = F.normalize(feats_B, dim=-1)

        if feats_A_ is None:
            if detach_grad:
                feats_A_ = feats_A.clone().detach()
            else:
                feats_A_ = feats_A
        else:
            feats_A_ = F.normalize(feats_A_, dim=-1)

        if feats_B_ is None:
            if detach_grad:
                feats_B_ = feats_B.clone().detach()
            else:
                feats_B_ = feats_B
        else:
            feats_B_ = F.normalize(feats_B_, dim=-1)

        # compute the kernel matrix --> patch2patch similarity matrix for both A and B # B, N, N
        kernel_matrix_A = feats_A @ feats_A_.transpose(1, 2)
        kernel_matrix_B = feats_B @ feats_B_.transpose(1, 2)

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
    def patch2patch_kernel_alignment_score_jsd(feats_A, feats_B, feats_A_=None, feats_B_=None, src_temp=1.0, tgt_temp=1.0, detach_grad=False):
        """
        Compute a patch-to-patch kernel alignment score using Jensen-Shannon Divergence.
        For each sample in the batch, we:
        1. Compute a patch-to-patch similarity matrix for feats_A and feats_B.
        2. Convert these similarity matrices to probability distributions via softmax.
        3. Compute the JSD between the corresponding distributions.
        
        The final returned score is 1 - JSD, so that:
            - Perfect alignment (identical distributions) -> JSD = 0, score = 1.0
            - Worst alignment (completely different distributions) -> JSD ~ 1, score ~ 0.0

        Args:
            feats_A: Tensor of shape (B, N, D)
            feats_B: Tensor of shape (B, N, E)
            feats_A_: (B x n_gpus, D) (if provided)
            feats_B_: (B x n_gpus, E) (if provided)
            src_temp: float, temperature for softmax applied to feats_A
            tgt_temp: float, temperature for softmax applied to feats_B
            detach_grad: bool, detach the gradient of the part of the feature matrix

        Returns:
            alignment_score: A scalar tensor representing the mean alignment score across the batch and all rows.
        """
        # Normalize feature vectors along the feature dimension
        feats_A = F.normalize(feats_A, dim=-1)
        feats_B = F.normalize(feats_B, dim=-1)

        if feats_A_ is None:
            if detach_grad:
                feats_A_ = feats_A.clone().detach()
            else:
                feats_A_ = feats_A
        else:
            feats_A_ = F.normalize(feats_A_, dim=-1)

        if feats_B_ is None:
            if detach_grad:
                feats_B_ = feats_B.clone().detach()
            else:
                feats_B_ = feats_B
        else:
            feats_B_ = F.normalize(feats_B_, dim=-1)

        # Compute patch-to-patch similarity matrices (B, N, N)
        kernel_matrix_A = feats_A @ feats_A_.transpose(1, 2)
        kernel_matrix_B = feats_B @ feats_B_.transpose(1, 2)

        # Convert similarities to probability distributions using softmax
        P = F.softmax(kernel_matrix_A / src_temp, dim=-1)  # (B, N, N)
        Q = F.softmax(kernel_matrix_B / tgt_temp, dim=-1)  # (B, N, N)

        # Compute the mixture distribution M = 0.5*(P+Q)
        M = 0.5 * (P + Q)

        # To compute KL divergences, ensure no log(0) by clamping
        eps = 1e-10
        P_clamped = torch.clamp(P, min=eps)
        Q_clamped = torch.clamp(Q, min=eps)
        M_clamped = torch.clamp(M, min=eps)

        # KL(P||M) = sum P * log(P/M) over the last dimension (N)
        KL_PM = torch.sum(P_clamped * (torch.log(P_clamped) - torch.log(M_clamped)), dim=-1)
        KL_QM = torch.sum(Q_clamped * (torch.log(Q_clamped) - torch.log(M_clamped)), dim=-1)

        # equivalent using F.kl_div
        # KL_PM = F.kl_div(M_clamped.log(), P_clamped, reduction='none', log_target=False).sum(dim=-1)  # (B, N)
        # KL_QM = F.kl_div(M_clamped.log(), Q_clamped, reduction='none', log_target=False).sum(dim=-1)  # (B, N)

        # Jensen-Shannon Divergence (base e)
        # JSD = 0.5 * (KL(P||M) + KL(Q||M))
        # To get JSD in [0,1], often we use log base 2:
        # JSD_base2 = JSD / log(2)
        # For compatibility, let’s convert to base 2:
        JSD = 0.5 * (KL_PM + KL_QM) / torch.log(torch.tensor(2.0))

        # JSD now is shape (B, N), average over all rows and batches
        JSD_mean = JSD.mean()

        # Convert JSD to a similarity score: perfect alignment (JSD=0) => score=1
        alignment_score = 1.0 - JSD_mean
        return alignment_score

    @staticmethod
    def sample2sample_kernel_alignment_score(feats_A, feats_B, feats_A_=None, feats_B_=None, detach_grad=False):
        """
        Compute the sample2sample kernel alignment score between two sets of features.
        Use a copy from the metrics.py to retain the gradient computation.
        feats_A: B, N, D
        feats_B: B, N, E
        feats_A_: (B x n_gpus, D) (if provided)
        feats_B_: (B x n_gpus, E) (if provided)
        detach_grad: bool, detach the gradient of the part of the feature matrix
        """
        # take the mean across last dimension # B, D
        feats_A = feats_A.mean(dim=-2)
        feats_B = feats_B.mean(dim=-2)

        # normalize the features along the last dimension
        feats_A = F.normalize(feats_A, dim=-1)
        feats_B = F.normalize(feats_B, dim=-1)

        if feats_A_ is None:
            if detach_grad:
                feats_A_ = feats_A.clone().detach()
            else:
                feats_A_ = feats_A
        else:
            feats_A_ = F.normalize(feats_A_, dim=-1)

        if feats_B_ is None:
            if detach_grad:
                feats_B_ = feats_B.clone().detach()
            else:
                feats_B_ = feats_B
        else:
            feats_B_ = F.normalize(feats_B_, dim=-1)

        # compute the kernel matrix --> sample2sample similarity matrix for both A and B # B, B
        kernel_matrix_A = feats_A @ feats_A_.transpose(0, 1)
        kernel_matrix_B = feats_B @ feats_B_.transpose(0, 1)

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

    @staticmethod
    def sample2sample_kernel_alignment_score_jsd(feats_A, feats_B, feats_A_=None, feats_B_=None, src_temp=1.0, tgt_temp=1.0, detach_grad=False):
        """
        Compute a sample-to-sample kernel alignment score using Jensen-Shannon Divergence.
        1. Average feats_A and feats_B across the N dimension to get (B, D) and (B, E).
        2. Normalize features along the last dimension.
        3. Compute sample-to-sample similarity matrices for A and B (B, B).
        4. Convert these matrices to probability distributions (softmax).
        5. Compute the JSD between these distributions.
        6. Convert JSD to an alignment score = 1 - JSD_mean, return as a scalar.

        Args:
            feats_A: (B, N, D)
            feats_B: (B, N, E)
            feats_A_: (B x n_gpus, D) (if provided)
            feats_B_: (B x n_gpus, E) (if provided)
            src_temp: float, temperature for softmax applied to feats_A
            tgt_temp: float, temperature for softmax applied to feats_B
            detach_grad: bool, detach the gradient of the part of the feature matrix

        Returns:
            alignment_score: a scalar float value.
        """
        # Average features across the N dimension: (B, N, D) -> (B, D)
        feats_A = feats_A.mean(dim=1)
        feats_B = feats_B.mean(dim=1)

        # Normalize the features
        feats_A = F.normalize(feats_A, dim=-1)
        feats_B = F.normalize(feats_B, dim=-1)

        if feats_A_ is None:
            if detach_grad:
                feats_A_ = feats_A.clone().detach()
            else:
                feats_A_ = feats_A
        else:
            feats_A_ = F.normalize(feats_A_, dim=-1)

        if feats_B_ is None:
            if detach_grad:
                feats_B_ = feats_B.clone().detach()
            else:
                feats_B_ = feats_B
        else:
            feats_B_ = F.normalize(feats_B_, dim=-1)

        # Compute similarity matrices (B, B)
        kernel_matrix_A = feats_A @ feats_A_.transpose(0, 1)
        kernel_matrix_B = feats_B @ feats_B_.transpose(0, 1)
        # print(kernel_matrix_A.shape, kernel_matrix_B.shape)

        # Convert similarities to probability distributions
        P = F.softmax(kernel_matrix_A / src_temp, dim=-1)  # (B, B)
        Q = F.softmax(kernel_matrix_B / tgt_temp, dim=-1)  # (B, B)
        M = 0.5 * (P + Q)  # Mixture distribution

        eps = 1e-10
        P_clamped = P.clamp(min=eps)
        Q_clamped = Q.clamp(min=eps)
        M_clamped = M.clamp(min=eps)

        # Compute KL divergences using F.kl_div
        # KL(P||M) = sum(P * log(P/M)) over dim=-1
        # Using F.kl_div: input=logM, target=P
        # logM = M_clamped.log()

        # Compute KL(P||M) and KL(Q||M) row-wise
        # KL(P||M) = sum_over_j P_j * log(P_j/M_j)
        KL_PM = (P_clamped * (P_clamped.log() - M_clamped.log())).sum(dim=-1)  # (B)
        KL_QM = (Q_clamped * (Q_clamped.log() - M_clamped.log())).sum(dim=-1)  # (B)

        # # equivalent using F.kl_div
        # KL_PM = F.kl_div(logM, P_clamped, reduction='none', log_target=False).sum(dim=-1)  # (B,)
        # KL_QM = F.kl_div(logM, Q_clamped, reduction='none', log_target=False).sum(dim=-1)  # (B,)

        # JSD in base 2
        log2 = torch.log(torch.tensor(2.0))
        JSD = 0.5 * (KL_PM + KL_QM) / log2  # (B,)

        # Mean JSD across batch
        JSD_mean = JSD.mean()

        # Alignment score: 1 - JSD
        alignment_score = 1.0 - JSD_mean
        return alignment_score

    def gather_dim(self, tensor, dim=0):
        """
        Gather the tensor across the specified dim dimension.
        """
        assert dim < tensor.dim()
        tensor_dim_first = tensor.permute(dim, *list(range(dim)), *list(range(dim+1, tensor.dim())))
        tensor_gathered = self.accelerator.gather(tensor_dim_first)
        tensor_gathered = tensor_gathered.permute(*list(range(1, dim+1)), 0, *list(range(dim+1, tensor_gathered.dim())))
        return tensor_gathered

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
        # NOTE: 
        # 1. `zs_tilde` is for REPA alignment, it must be after projection layer
        # 2. `fs_tilde` is for kernel alignment, it can be before or after projection layer depending on `ka_after_proj`
        model_output, zs_tilde_layers, fs_tilde_layers, unprojected_detached_feats, all_layer_feats = model(
            model_input, time_input.flatten(), ka_use_projection=alignment_kwargs["ka_aft_proj"],
            return_all_layers=alignment_kwargs["compute_alignment"] and alignment_kwargs["max_score_across_layers"],
            **model_kwargs
        )
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection and kernel alignment loss
        proj_loss, kernel_alignment_loss = 0., 0.

        bsz = zs[0].shape[0]
        # REPA loss
        proj_loss = 0.
        for zs_tilde in zs_tilde_layers:
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz * len(zs_tilde_layers))


        # Compute the kernel alignment loss
        if self.loss_type is None:
            # If no kernel_alignment_loss is given, then set it to 0 and move to corresponding device
            kernel_alignment_loss = 0.
            kernel_alignment_loss = torch.tensor(kernel_alignment_loss, device=zs[0].device)

        elif self.loss_type == "patch2patch":
            # NOTE: We should compute kernel alignment with unprojected features only
            for fs_tilde in fs_tilde_layers:
                for i, (z, f_tilde) in enumerate(zip(zs, fs_tilde)):
                    # NOTE: The loss should be the negative of the alignment score (minimize the negative alignment score)
                    kernel_alignment_loss += -self.patch2patch_kernel_alignment_score(z, f_tilde, detach_grad=alignment_kwargs["ka_detach_grad"])

        elif self.loss_type == "patch2patch_jsd":
            # NOTE: JSD requires the temperature arugmnet
            for fs_tilde in fs_tilde_layers:
                for i, (z, f_tilde) in enumerate(zip(zs, fs_tilde)):
                    kernel_alignment_loss += -self.patch2patch_kernel_alignment_score_jsd(
                        z,
                        f_tilde,
                        src_temp=alignment_kwargs["p2p_jsd_src_temp"],
                        tgt_temp=alignment_kwargs["p2p_jsd_tgt_temp"],
                        detach_grad=alignment_kwargs["ka_detach_grad"],
                    )

        elif "sample2sample" in self.loss_type:
            # For sample2sample, we need to gather all features to compute the alignment score

            # Get a copy of detached fs_tilde and gather, L -> multiple alignment layers, P -> num_projectors
            fs_tilde_stacked = torch.stack([torch.stack(fs_tilde_layer) for fs_tilde_layer in fs_tilde_layers]).detach()  # [L, P, B, N, D]
            fs_tilde_mean_reduced = fs_tilde_stacked.mean(dim=3)  # [L, P, B, D]
            fs_tilde_gathered = self.gather_dim(fs_tilde_mean_reduced, dim=2)  # [L, P, B x n_gpus, D]

            # Get a copy of projector (e.g. DINOv2) features (already detached)
            zs_stacked = torch.stack(zs)  # [n_projs, B, N, D]
            zs_mean_reduced = zs_stacked.mean(dim=2)  # [n_projs, B, D]
            zs_gathered = self.gather_dim(zs_mean_reduced, dim=1)  # [n_projs, B x n_gpus, D]

            if self.loss_type == "sample2sample":
                # NOTE We should compute kernel alignment with unprojected features only
                for fs_tilde, fs_tilde_g in zip(fs_tilde_layers, fs_tilde_gathered):
                    for i, (z, z_g, f_tilde, f_tilde_g) in enumerate(zip(zs, zs_gathered, fs_tilde, fs_tilde_g)):
                        # NOTE: The loss should be the negative of the alignment score (minimize the negative alignment score)
                        kernel_alignment_loss += -self.sample2sample_kernel_alignment_score(
                            feats_A=z,  # [B, N, D]
                            feats_B=f_tilde,  # [B, N, D]
                            feats_A_=z_g,  # [B x n_gpus, D]
                            feats_B_=f_tilde_g,  # [B x n_gpus, D]
                            detach_grad=alignment_kwargs["ka_detach_grad"],
                        )

            elif self.loss_type == "sample2sample_jsd":
                for fs_tilde, fs_tilde_g in zip(fs_tilde_layers, fs_tilde_gathered):
                    for i, (z, z_g, f_tilde, f_tilde_g) in enumerate(zip(zs, zs_gathered, fs_tilde, fs_tilde_g)):
                        # print(z.shape, f_tilde.shape, z_g.shape, f_tilde_g.shape)
                        kernel_alignment_loss += -self.sample2sample_kernel_alignment_score_jsd(
                            feats_A=z,  # [B, N, D]
                            feats_B=f_tilde, # [B, N, D]
                            feats_A_=z_g,       # [B x n_gpus, D]
                            feats_B_=f_tilde_g,  # [B x n_gpus, D]
                            src_temp=alignment_kwargs["s2s_jsd_src_temp"],
                            tgt_temp=alignment_kwargs["s2s_jsd_tgt_temp"],
                            detach_grad=alignment_kwargs["ka_detach_grad"],
                        )

            else:
                raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        # Normalize the kernel alignment loss based on the num_projs and num_align_layers
        kernel_alignment_loss /= len(zs) * len(fs_tilde_layers)

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
                                    # jsd needs temperature, use the source temperature for logging
                                    measure_kwargs["temperature"] = alignment_kwargs["p2p_jsd_src_temp"]
                                elif metric == "sample2sample_kernel_alignment_score_jsd":
                                    # jsd needs temperature, use the source temperature for logging
                                    measure_kwargs["temperature"] = alignment_kwargs["s2s_jsd_src_temp"]

                                # FIXME: for now we have some problems with the sample2sample alignment score computation
                                #        we are not gathering all features to compute sample2sample self-correlation for logging, which means we are
                                #        compute the correlation within the batch on the current GPU, but we leave it here for now
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
                        assert len(unprojected_detached_feats) > 0, "There should be at least one layer that is aligned with both REPA and kernel alignment..."
                        unprojected_detached_feat = unprojected_detached_feats[-1]  # Let's use the last layer to evaluate alignment
                        for z, feat_tilde in zip(zs, unprojected_detached_feat):
                            # NOTE: For the CKNNA score, we take the mean across patches, to get a single feature vector for each sample
                            measure_kwargs = {}
                            if "kernel_alignment" in metric:
                                # Our method assumes [B, L, D]
                                feats_A = z
                                feats_B = feat_tilde
                            else:
                                # Rep paper method assumes [B, D]
                                feats_A = z.mean(dim=1)
                                feats_B = feat_tilde.mean(dim=1)

                            if metric == "cknna":
                                # cknna needs topk
                                measure_kwargs["topk"] = alignment_kwargs["cknna_topk"]
                            elif metric == "patch2patch_kernel_alignment_score_jsd":
                                # jsd needs temperature, use the source temperature for logging
                                measure_kwargs["temperature"] = alignment_kwargs["p2p_jsd_src_temp"]
                            elif metric == "sample2sample_kernel_alignment_score_jsd":
                                # jsd needs temperature, use the source temperature for logging
                                measure_kwargs["temperature"] = alignment_kwargs["s2s_jsd_src_temp"]

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
