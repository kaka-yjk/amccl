import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger('MSA')

class contrastive_loss(nn.Module):
    def __init__(self, dataset_name, device, k_threshold, gain, temperature, lambda_refine, alpha,
             inter_weight, intra_weight, gamma, seed=None):
        super(contrastive_loss, self).__init__()
        self.dataset_name = dataset_name
        self.device = device
        self.k_threshold = k_threshold
        self.gain = gain
        self.temperature = temperature
        self.lambda_refine = lambda_refine
        self.alpha = alpha
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight
        self.gamma = gamma
        self.seed = seed
        self.ref_file_path = 'results/results/ref.txt'
        self.gamma_file_path = 'results/results/gamma.txt'
        try:
            os.makedirs(os.path.dirname(self.ref_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.gamma_file_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory for {self.ref_file_path} or {self.gamma_file_path}: {e}")
            raise

    def inter_modal_loss(self, input1, input2):
        sim_matrix = torch.matmul(input1, input2.T)
        pos_result = self.get_positive_pair(sim_matrix)
        neg_result = self.get_negative_pair(sim_matrix)
        pos_result = torch.exp(pos_result / self.temperature)
        neg_result = torch.exp(neg_result / self.temperature)
        return pos_result, neg_result

    def get_positive_pair(self, sim_matrix, need_supervised=True):
        if not need_supervised:
            n = sim_matrix.shape[1]
            diag = torch.eye(n, device=self.device)
            result = sim_matrix * diag
        else:
            result = sim_matrix * self.get_supervised_mask(self.label_map, pair='positive')
        logger.debug(f"get_positive_pair result type: {result.dtype}, result shape: {result.shape}")
        return result.float()

    def get_negative_pair(self, sim_matrix, need_supervised=True):
        if not need_supervised:
            n = sim_matrix.shape[1]
            diag = torch.eye(n, device=self.device)
            result = sim_matrix * (1 - diag) * 0.8
        else:
            result = sim_matrix * self.get_supervised_mask(self.label_map, pair='negative') * 0.8
        logger.debug(f"get_negative_pair result type: {result.dtype}, result shape: {result.shape}")
        return result.float()

    def get_supervised_mask(self, label_map, pair='positive'):
        logger.debug(f"label_map type: {label_map.dtype}, label_map shape: {label_map.shape}, label_map: {label_map}")
        if label_map.dim() == 1:
            label_map = label_map.unsqueeze(0)
        elif label_map.dim() > 2:
            label_map = label_map.squeeze()
            if label_map.dim() == 1:
                label_map = label_map.unsqueeze(0)

        if torch.isnan(label_map).any() or torch.isinf(label_map).any():
            logger.warning(f"label_map contains NaN or Inf values: {label_map}")
            label_map = torch.where(torch.isnan(label_map) | torch.isinf(label_map), torch.zeros_like(label_map),
                                    label_map)

        new_label_map = self.label_map_reconstruction(label_map)
        label_map_length = label_map.shape[1]
        self_supervised_mask = torch.zeros(label_map_length, label_map_length, device=self.device, dtype=torch.float)

        intensity_i = new_label_map.unsqueeze(2).expand(-1, -1, label_map_length)
        intensity_j = new_label_map.unsqueeze(1).expand(-1, label_map_length, -1)
        intensity_i = intensity_i.squeeze(0)
        intensity_j = intensity_j.squeeze(0)
        intensity_diff = torch.abs(intensity_i - intensity_j)

        if torch.isnan(intensity_diff).any() or torch.isinf(intensity_diff).any():
            logger.warning(f"intensity_diff contains NaN or Inf values: {intensity_diff}")
            intensity_diff = torch.where(torch.isnan(intensity_diff) | torch.isinf(intensity_diff),
                                         torch.zeros_like(intensity_diff), intensity_diff)

        polarity_i = torch.sign(intensity_i)
        polarity_j = torch.sign(intensity_j)

        if pair == 'positive':
            intensity_condition = intensity_diff <= self.k_threshold
            polarity_condition = polarity_i == polarity_j
            pos_condition = intensity_condition & polarity_condition
            if not pos_condition.any():
                logger.warning(f"No positive pairs found with k_threshold={self.k_threshold}")
            self_supervised_mask[pos_condition] = self.gain * torch.exp(
                -intensity_diff[pos_condition] / self.k_threshold)
        else:
            intensity_condition = intensity_diff > self.k_threshold
            polarity_condition = polarity_i != polarity_j
            neg_condition1 = intensity_condition
            neg_condition2 = (intensity_diff <= self.k_threshold) & polarity_condition
            neg_condition = neg_condition1 | neg_condition2
            self_supervised_mask[neg_condition] = self.gain * torch.exp(
                (intensity_diff[neg_condition] - self.k_threshold) / self.k_threshold)

        logger.debug(
            f"self_supervised_mask type: {self_supervised_mask.dtype}, mask shape: {self_supervised_mask.shape}")
        return self_supervised_mask.float()

    def label_map_reconstruction(self, label_map):
        if self.dataset_name in ['mosi', 'mosei']:
            new_label_map = (label_map + 3) / 3 - 1  # MOSI 和 MOSEI: [-3, 3] -> [-1, 1]
        elif self.dataset_name in ['sims', 'simsv2']:
            new_label_map = label_map
        else:
            new_label_map = label_map
        logger.debug(
            f"new_label_map type: {new_label_map.dtype}, new_label_map shape: {new_label_map.shape}, new_label_map: {new_label_map}")
        if torch.isnan(new_label_map).any() or torch.isinf(new_label_map).any():
            logger.warning(f"new_label_map contains NaN or Inf values: {new_label_map}")
            new_label_map = torch.where(torch.isnan(new_label_map) | torch.isinf(new_label_map),
                                        torch.zeros_like(new_label_map), new_label_map)
        return new_label_map

    def compute_contrastive_loss(self, inter_pos_result, inter_neg_result, intra_pos_result=None,
                                 intra_neg_result=None):
        zeros = torch.zeros_like(inter_pos_result)
        molecular = inter_pos_result + zeros
        denominator = inter_pos_result + zeros + inter_neg_result + zeros
        return -torch.log(torch.div(molecular.sum(1), denominator.sum(1))).mean()

    def compute_improved_inter_modal_loss(self, input1, input2):
        sim_matrix = torch.matmul(input1, input2.T)
        pos_result = self.get_positive_pair(sim_matrix)
        pos_exp = torch.exp(pos_result / self.temperature)
        neg_sum = torch.sum(torch.exp(sim_matrix / self.temperature), dim=1, keepdim=True)
        contrastive_term = -torch.log(pos_exp / neg_sum).mean()
        refine_term = -pos_result.mean()

        return contrastive_term + self.lambda_refine * refine_term, refine_term

    def compute_improved_intra_modal_loss(self, input1):
        sim_matrix = torch.matmul(input1, input1.T)
        pos_result = self.get_positive_pair(sim_matrix)
        pos_exp = torch.exp(pos_result / self.temperature)
        neg_sum = torch.sum(torch.exp(sim_matrix / self.temperature), dim=1, keepdim=True)
        contrastive_term = -torch.log(pos_exp / neg_sum).mean()
        refine_term = -pos_result.mean()

        return contrastive_term + self.lambda_refine * refine_term, refine_term

    def compute_margin_loss(self, input1, input2):
        sim_inter = torch.matmul(input1, input2.T).float()
        pos_inter = self.get_positive_pair(sim_inter)
        if pos_inter.numel() == 0 or pos_inter.dtype != torch.float or torch.isnan(pos_inter).any():
            logger.warning(
                f"pos_inter is empty, not float, or contains NaN: numel={pos_inter.numel()}, dtype={pos_inter.dtype}")
            s_inter = torch.tensor(0.0, device=self.device, dtype=torch.float)
        else:
            s_inter = pos_inter.mean().float()
        logger.debug(f"s_inter type: {s_inter.dtype}, s_inter: {s_inter}")

        sim_intra1 = torch.matmul(input1, input1.T).float()
        pos_intra1 = self.get_positive_pair(sim_intra1)
        if pos_intra1.numel() == 0 or pos_intra1.dtype != torch.float or torch.isnan(pos_intra1).any():
            logger.warning(
                f"pos_intra1 is empty, not float, or contains NaN: numel={pos_intra1.numel()}, dtype={pos_intra1.dtype}")
            s_intra1 = torch.tensor(0.0, device=self.device, dtype=torch.float)
        else:
            s_intra1 = pos_intra1.mean().float()
        logger.debug(f"s_intra1 type: {s_intra1.dtype}, s_intra1: {s_intra1}")

        sim_intra2 = torch.matmul(input2, input2.T).float()
        pos_intra2 = self.get_positive_pair(sim_intra2)
        if pos_intra2.numel() == 0 or pos_intra2.dtype != torch.float or torch.isnan(pos_intra2).any():
            logger.warning(
                f"pos_intra2 is empty, not float, or contains NaN: numel={pos_intra2.numel()}, dtype={pos_intra2.dtype}")
            s_intra2 = torch.tensor(0.0, device=self.device, dtype=torch.float)
        else:
            s_intra2 = pos_intra2.mean().float()
        logger.debug(f"s_intra2 type: {s_intra2.dtype}, s_intra2: {s_intra2}")

        s_intra = (s_intra1 + s_intra2) / 2
        logger.debug(f"s_intra type: {s_intra.dtype}, s_intra: {s_intra}")

        margin_loss = torch.relu(self.alpha - (s_intra - s_inter))
        logger.debug(f"margin_loss: {margin_loss}")

        return margin_loss

    def compute_single_loss(self, input1, input2):
        logger.debug(f"compute_single_loss input1 type: {input1.dtype}, input1 shape: {input1.shape}")
        logger.debug(f"compute_single_loss input2 type: {input2.dtype}, input2 shape: {input2.shape}")
        input1 = F.normalize(input1, dim=1)
        input2 = F.normalize(input2, dim=1)

        improved_inter_loss, refine_term = self.compute_improved_inter_modal_loss(input1, input2)
        margin_loss = self.compute_margin_loss(input1, input2)

        return improved_inter_loss + margin_loss, refine_term, margin_loss #margin_loss权重设置为1

    def forward(self, outputs, label_map, task_loss=None):
        self.label_map = label_map
        norm_v = F.normalize(outputs['Feature_v'], dim=1)
        norm_a = F.normalize(outputs['Feature_a'], dim=1)
        norm_t = F.normalize(outputs['Feature_t'], dim=1)


        valoss, va_refine_term, va_margin_loss = self.compute_single_loss(outputs['Feature_v'], outputs['Feature_a'])
        vtloss, vt_refine_term, vt_margin_loss = self.compute_single_loss(outputs['Feature_v'], outputs['Feature_t'])
        taloss, ta_refine_term, ta_margin_loss = self.compute_single_loss(outputs['Feature_t'], outputs['Feature_a'])


        intra_loss_v, intra_refine_term_v = self.compute_improved_intra_modal_loss(norm_v)
        intra_loss_a, intra_refine_term_a = self.compute_improved_intra_modal_loss(norm_a)
        intra_loss_t, intra_refine_term_t = self.compute_improved_intra_modal_loss(norm_t)


        inter_loss = valoss + vtloss + taloss
        intra_loss = intra_loss_v + intra_loss_a + intra_loss_t
        contrastive_loss = self.inter_weight * inter_loss + self.intra_weight * intra_loss

        return contrastive_loss