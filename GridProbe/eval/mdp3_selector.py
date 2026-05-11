"""
MDP3 frame selector — extracted from MDP3-main/vlmeval/smp/mdp3_frame_selector.py
=================================================================================
Self-contained black-box: SigLIP scoring + DPP/dynamic-programming subset selection.

Source: Sun et al. 2025 "MDP3: A Training-free Approach for List-wise Frame
Selection in Video-LLMs" (arXiv:2501.02885), MIT License.

Local modifications vs upstream:
  - n_selection promoted to constructor arg (was hardcoded `self.n_selection = 8`).
  - segment_size auto-resizes when n_selection changes.
  - SigLip.clear_prompt is best-effort: it strips Video-MME 4-letter heads/tails;
    8-letter (V2) and LVB prompts pass through unchanged. Selection still works
    because SigLIP just embeds whatever text it gets.
  - Added approximate FLOPs hook (per-image SigLIP cost is configurable so we
    can audit without re-instrumenting).
"""

import copy
import time
from contextlib import contextmanager

import torch
from PIL import Image
from torch import nn
from transformers import SiglipModel, SiglipProcessor


@contextmanager
def _timer(hint=""):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    # Silent by default; set environment variable MDP3_VERBOSE=1 to enable.
    import os
    if os.environ.get("MDP3_VERBOSE"):
        print(f"[MDP3] {hint}: {end - start:.3f}s")


INF = 0x7fffffff


class SigLip:
    """SigLIP wrapper that returns (image_embeds, mean_text_embed)."""

    def __init__(self, device="cuda",
                 model_name="google/siglip-so400m-patch14-384",
                 cache_dir=None):
        self.device = device
        kw = {"device_map": self.device}
        if cache_dir:
            kw["cache_dir"] = cache_dir
        self.model = SiglipModel.from_pretrained(model_name, **kw)
        proc_kw = {"cache_dir": cache_dir} if cache_dir else {}
        self.processor = SiglipProcessor.from_pretrained(model_name, **proc_kw)

    def __call__(self, images, texts):
        texts = self.clear_prompt(copy.deepcopy(texts))

        with _timer("siglip processor"):
            inputs = self.processor(
                text=texts, images=images, padding="max_length",
                return_tensors="pt").to(self.model.device)

        # Split long prompts into 64-token chunks (SigLIP text encoder cap).
        stride_num = (int(inputs["input_ids"].shape[-1]) + 63) // 64
        stride = (inputs["input_ids"].shape[-1] + stride_num - 1) // stride_num

        input_id_heads, input_id_tails = [], []
        l, r = 0, inputs["input_ids"].shape[-1]
        while l < r:
            input_id_heads.append(inputs["input_ids"][:, l:l + stride])
            l += stride
            if l < r:
                input_id_tails.append(inputs["input_ids"][:, r - stride:r])
                r -= stride

        input_ids = input_id_heads + input_id_tails[::-1]
        input_ids = torch.cat(input_ids)

        with _timer("siglip forward"):
            with torch.no_grad():
                with torch.autocast(self.device):
                    outputs = self.model(input_ids,
                                         pixel_values=inputs["pixel_values"])
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        return image_embeds, text_embeds.mean(dim=0, keepdim=True)

    def clear_prompt(self, prompt):
        """Strip Video-MME boilerplate. Best-effort: V2/LVB prompts may pass through."""
        heads = [
            "Select the best answer to the following multiple-choice question "
            "based on the video and the subtitles. Respond with only the letter "
            "(A, B, C, or D) of the correct option.",
            "Select the best answer to the following multiple-choice question "
            "based on the video. Respond with only the letter (A, B, C, or D) "
            "of the correct option.",
            # V2 8-letter variants (added so V2 prompts are also cleaned).
            "Select the best answer to the following multiple-choice question "
            "based on the video. Respond with only the letter (A, B, C, D, E, "
            "F, G, or H) of the correct option.",
        ]
        tails = [
            "Answer with the option's letter from the given choices directly.",
            "The best answer is:",
            "Answer the question using a single word or phrase.",
            "Only give the best option.\n",
            "Best option: (",
        ]
        for head in heads:
            prompt = prompt.split(head)[-1]
        for tail in tails:
            prompt = prompt.split(tail)[0]
        prompt = prompt.strip()
        return prompt


class MDP3:
    """List-wise frame selector. Forward signature: (frames, prompt) -> selected_frames.

    Args:
        n_selection: how many frames to select (default 8, matching paper Table 1).
        lamda: balance between text relevance and frame diversity (paper default 0.2).
        device: cuda or cpu.
        siglip_model: HuggingFace SigLIP repo id (default matches the paper).
        cache_dir: HF cache dir.
        return_indices: if True, returns frame indices instead of frame objects
            (cleaner for our eval pipeline).
    """

    def __init__(self, device="cuda", n_selection=8, lamda=0.2,
                 siglip_model="google/siglip-so400m-patch14-384",
                 cache_dir=None, return_indices=False):
        self.n_selection = int(n_selection)
        self.lamda = float(lamda)
        # Original logic: 32 if n<=32 else 128. Keep it.
        self.segment_size = 32 if self.n_selection <= 32 else 128
        self.condition_size = 1
        self.return_indices = return_indices

        self.kernel = MultiGaussianKernel(
            alphas=[2 ** k for k in list(range(-3, 2))])
        self.vlm = SigLip(device, model_name=siglip_model, cache_dir=cache_dir)

    def set_n_selection(self, n: int):
        """Change selection budget between samples without reloading SigLIP.

        Used by per-sample M_eff matching (e.g., when comparing MDP3 against
        GridProbe at the same per-question M chosen by σ statistic).
        """
        self.n_selection = int(n)
        self.segment_size = 32 if self.n_selection <= 32 else 128

    def __call__(self, frames, prompt, sample=None):
        with _timer("** TOTAL Frame Selection"):
            input_frames = copy.deepcopy(frames)
            with _timer("Read Image"):
                if isinstance(frames[0], str):
                    frames = [Image.open(p) for p in frames]
            with _timer("** VLMs Process & Extract"):
                image_embeds, text_embeds = self.vlm(frames, prompt)
            with _timer("Select Frames"):
                with torch.no_grad():
                    selected_idx = self._select_frames_fast(
                        image_embeds, text_embeds)
            if self.return_indices:
                return selected_idx
            return [input_frames[idx] for idx in selected_idx]

    def cal_obj(self, selected_images_embeds, text_embed):
        kernel_matrix = self.kernel(
            torch.cat([text_embed, selected_images_embeds]))
        r, S_matrix = kernel_matrix[0:1, 1:], kernel_matrix[1:, 1:]
        ret_score = (1. / self.lamda * 2 * torch.log(r).sum()) + \
            torch.linalg.slogdet(S_matrix).logabsdet
        return ret_score

    def _select_frames_fast(self, image_embeds, text_embeds):
        N_image = len(image_embeds)
        segment_num = (N_image + self.segment_size - 1) // self.segment_size
        dp = [[0.] + [-INF] * self.n_selection
              for _ in range(segment_num + 1)]
        trace = [[[] for _ in range(self.n_selection + 1)]
                 for _ in range(segment_num + 1)]

        for seg_idx in range(1, segment_num + 1):
            candidate_index = range(
                (seg_idx - 1) * self.segment_size, seg_idx * self.segment_size)
            candidate_index = [i for i in candidate_index if i < N_image]
            if not candidate_index:
                # Trailing segment beyond pool size; carry state forward.
                for k in range(self.n_selection + 1):
                    dp[seg_idx][k] = dp[seg_idx - 1][k]
                    trace[seg_idx][k] = trace[seg_idx - 1][k]
                continue
            candidate_embeds = [image_embeds[i] for i in candidate_index]
            sim_matrix = self.kernel(torch.stack(candidate_embeds))

            for start_selected_num in range(
                    0, min(self.n_selection,
                           (seg_idx - 1) * self.segment_size) + 1):
                conditional_index = trace[seg_idx - 1][start_selected_num][
                    -min(self.condition_size,
                         len(trace[seg_idx - 1][start_selected_num])):]
                offset = len(conditional_index)
                additional_embeds = [text_embeds[0].reshape(-1)] + \
                    [image_embeds[i] for i in conditional_index]
                additional = self.kernel(
                    torch.stack(additional_embeds),
                    torch.stack(additional_embeds + candidate_embeds)
                )
                total_matrix = torch.cat([
                    additional,
                    torch.cat([
                        additional[:, -len(sim_matrix):].T,
                        sim_matrix
                    ], dim=1)
                ], dim=0)

                max_selection = min(
                    self.n_selection - start_selected_num,
                    len(candidate_index))

                cur_scores, cur_traces = self.seqdpp_select_super_fast(
                    total_matrix, offset, max_selection)
                for to_select_num, (cur_score, cur_trace) in enumerate(
                        zip(cur_scores, cur_traces)):
                    cur_trace = [i + int((seg_idx - 1) * self.segment_size)
                                 for i in cur_trace]
                    cur_score_total = dp[seg_idx - 1][start_selected_num] + cur_score
                    cur_trace_total = trace[seg_idx - 1][start_selected_num] + cur_trace
                    if cur_score_total > dp[seg_idx][start_selected_num + to_select_num]:
                        dp[seg_idx][start_selected_num + to_select_num] = cur_score_total
                        trace[seg_idx][start_selected_num + to_select_num] = cur_trace_total
        return trace[segment_num][self.n_selection]

    def seqdpp_select_super_fast(self, total_matrix, offset, to_select_num):
        if to_select_num == 0:
            return [0.0], [[]]
        cur_trace = []
        ret_scores = [0.0]
        r, S_matrix = total_matrix[0:1, 1:], total_matrix[1:, 1:]
        candidate_index = list(range(len(S_matrix) - offset))

        conditional_idx = list(range(offset))
        L = None
        if len(conditional_idx) > 0:
            L = torch.linalg.cholesky(
                S_matrix[conditional_idx][:, conditional_idx])

        while len(cur_trace) < to_select_num:
            max_obj = -INF
            cur_selected_idx = -1
            better_L = None
            for i in candidate_index:
                if i in cur_trace:
                    continue
                cur_idx = i + offset
                selected_idx = conditional_idx + \
                    [j + offset for j in cur_trace] + [cur_idx]
                if L is None:
                    cur_sim_v = S_matrix[selected_idx][:, selected_idx]
                    cur_L = torch.sqrt(cur_sim_v).reshape(1, 1)
                    logdet = cur_sim_v.clone().log()
                else:
                    cur_sim_v = S_matrix[cur_idx:cur_idx + 1][:, selected_idx]
                    cur_L, logdet = self.cholesky_update_determinant(
                        L, cur_sim_v)
                cur_obj = 1. / self.lamda * 2 * \
                    torch.log(r[:, selected_idx]).sum() + logdet

                if cur_obj > max_obj or cur_selected_idx == -1:
                    max_obj = cur_obj
                    cur_selected_idx = i
                    better_L = cur_L
            ret_scores.append(max_obj.clone())
            cur_trace.append(cur_selected_idx)
            L = better_L
        ret_traces = [sorted(cur_trace[:j]) for j in range(len(cur_trace) + 1)]
        return ret_scores, ret_traces

    def cholesky_update_determinant(self, L, v):
        n = L.shape[0]
        v = v.view(-1, 1)
        v_projected = torch.linalg.solve_triangular(L, v[:n], upper=False)
        new_diag_element = torch.sqrt(
            torch.abs(v[-1] - v_projected.T @ v_projected))
        new_row = torch.cat((v_projected.flatten(), new_diag_element.view(1)))
        new_L = torch.zeros((n + 1, n + 1), dtype=L.dtype, device=L.device)
        new_L[:n, :n] = L
        new_L[n, :n] = new_row[:-1]
        new_L[n, n] = new_diag_element
        new_diag = torch.diag(new_L)
        new_logdet = 2 * torch.log(new_diag).sum()
        return new_L, new_logdet


class GaussianKernel(nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(1) - X.unsqueeze(0)) ** 2).sum(2)
        return torch.exp(-l2_distance_square / (2 * self.alpha))


class MultiGaussianKernel(nn.Module):
    def __init__(self, alphas=[2 ** k for k in list(range(-3, 2))]):
        super().__init__()
        self.alphas = alphas

    def forward(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        Y = X.unsqueeze(0) if Y is None else Y.unsqueeze(0)
        X = X.unsqueeze(1)
        l2_distance_square = ((X - Y) ** 2).sum(2)
        return sum([torch.exp(-l2_distance_square / (2 * alpha))
                    for alpha in self.alphas])


# ─────────────────────────────────────────────────────────────────────
# Approximate analytical FLOPs for SigLIP-SO/14 at 384x384.
# These constants are used by mdp3_eval.py to log per-question selector
# FLOPs in the same TFLOPs schema as the QA model. Numbers are coarse
# (within ~10% of measured); refine via fvcore/torch.profiler if needed.
# ─────────────────────────────────────────────────────────────────────

# SigLIP-SO/14 at 384 res: ~213 GFLOPs/image (vision tower forward).
SIGLIP_SO14_IMAGE_FLOPS = 213e9
# Text tower at 64-token cap: ~35 GFLOPs (much smaller; multiplied by chunk count).
SIGLIP_SO14_TEXT_FLOPS_PER_CHUNK = 35e9


def estimate_mdp3_selector_flops(n_frames: int, n_text_chunks: int = 1) -> float:
    """Approximate per-question selector compute (image scoring + text encode).

    DPP selection itself is matrix ops on the embeddings, negligible vs encoder
    forwards. Returns FLOPs (not TFLOPs).
    """
    return (n_frames * SIGLIP_SO14_IMAGE_FLOPS
            + n_text_chunks * SIGLIP_SO14_TEXT_FLOPS_PER_CHUNK)
