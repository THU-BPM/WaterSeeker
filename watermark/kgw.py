# ============================================
# kgw.py
# Description: Implementation of KGW algorithm
# ============================================

import json
import time
import torch
import numpy as np
from math import sqrt
from functools import partial
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList

class KGWConfig:
    """Config class for KGW algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/KGW.json')
        else:
            config_dict = load_config_file(algorithm_config)

        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.hash_key = config_dict['hash_key']
        self.z_threshold = config_dict['z_threshold']
        self.prefix_length = config_dict['prefix_length']

        with open('threshold_dict/z_threshold_dict_1e6.json', 'r') as f:
            self.z_threshold_dict = json.load(f)

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class KGWUtils:
    """Utility class for KGW algorithm, contains helper functions."""

    def __init__(self, config: KGWConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW utility class.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
    
    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the RNG with the last min_prefix_len tokens of the input_ids."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return
    
    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        self._seed_rng(input_ids)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids

    def _compute_z_score(self, observed_count: int, T: int) -> float: 
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z

    def _precompute_greenlists(self, input_ids: torch.Tensor) -> list:
        greenlists = [None for _ in range(len(input_ids))]
        for i in range(self.config.prefix_length, len(input_ids)):
            greenlists[i] = self.get_greenlist_ids(input_ids[:i])
        return greenlists
    
    def score_sequence_by_window(self, token_flags: list, start_idx: int, end_idx: int) -> float:
        """Compute z-score for a window of length L starting at start_idx."""
        green_token_count = sum(token_flags[start_idx: end_idx])
        z_score = self._compute_z_score(green_token_count, end_idx - start_idx)
        return z_score
    
    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Compute z-score for the input_ids."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
                )
            )   
        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags

    def detect_anomalies(self, token_flags, window_size=50, threshold_1=0.5, threshold_2=1.5, top_k=20, min_length=100, tolerance=100):
        """Detect anomalies in the token flags."""
        
        # Compute the proportion of green tokens in each window
        proportions = []
        for i in range(len(token_flags) - window_size + 1):
            window = token_flags[i:i + window_size]
            proportion = sum(window) / window_size
            proportions.append(proportion)
        
        # Compute the mean and standard deviation of the proportions
        mean_proportion = np.mean(proportions)
        std_proportion = np.std(proportions)

        # Find the top-k proportions and calculate the top-mean proportion
        top_proportions = sorted(proportions, reverse=True)[:top_k]
        top_mean_proportion = np.mean(top_proportions)

        # Calculate the difference value
        diff_value = max((top_mean_proportion - mean_proportion) * threshold_1, std_proportion * threshold_2)

         # Find the anomalies
        anomalies = [i for i, p in enumerate(proportions) if p > mean_proportion + diff_value]

        # Merge adjacent anomalies
        merged_anomalies = []
        current_segment = []

        for i in range(len(anomalies)):
            if not current_segment:
                current_segment = [anomalies[i]]
            else:
                if anomalies[i] - current_segment[-1] <= tolerance:
                    current_segment.append(anomalies[i])
                else:
                    merged_anomalies.append(current_segment)
                    current_segment = [anomalies[i]]
        
        if current_segment:
            merged_anomalies.append(current_segment)

        valid_segments = []

        # Filter out the segments that are too short
        for segment in merged_anomalies:
            if (min_length <= (segment[-1] - segment[0] + window_size - 1)):
                valid_segments.append((segment[0], segment[-1] + window_size - 1))

        # 返回异常片段的开始和结束位置
        if valid_segments:
            return valid_segments
        else:
            return None


class KGWLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW algorithm, process logits to add watermark."""

    def __init__(self, config: KGWConfig, utils: KGWUtils, *args, **kwargs) -> None:
        """
            Initialize the KGW logits processor.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
                utils (KGWUtils): Utility class for the KGW algorithm.
        """
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        return scores
    

class KGW:
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = KGWConfig(algorithm_config, transformers_config)
        self.utils = KGWUtils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils)
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Full-text Detection."""
        
        # Encode the text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute z_score
        z_score, _ = self.utils.score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold_dict[str(len(encoded_text) - self.config.prefix_length) if len(encoded_text) - self.config.prefix_length <= 1000 else '1000']

        # Return results based on the return_dict flag
        if return_dict:
            if is_watermarked:
                return {"is_watermarked": is_watermarked, "indices": [(0, len(encoded_text), z_score)]}
            else:
                return {"is_watermarked": is_watermarked, "indices": []}
    
    def get_token_flags(self, text: str, *args, **kwargs):
        """Get token flags for the text."""
        
        # Encode the text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute token flags
        _, token_flags = self.utils.score_sequence(encoded_text)

        return token_flags
    
    def detect_watermark_win_max(self, text:str, return_dict: bool = True, min_L: int = 1, max_L=None, *args, **kwargs):
        """Detect watermark in the text using WinMax algorithm."""
        
        # Get token flags
        token_flags = self.get_token_flags(text)

        if max_L is None:
            max_L = len(token_flags) - self.config.prefix_length

        max_z_score = float('-inf')
        flag_start_idx, flag_end_idx = -1, -1

        # Traverse all possible windows
        for L in range(min_L, max_L + 1):
            for start_idx in range(self.config.prefix_length, len(token_flags) - L + 1):
                z_score = self.utils.score_sequence_by_window(token_flags, start_idx, start_idx + L)
                if z_score > max_z_score:
                    max_z_score = z_score
                    flag_start_idx, flag_end_idx = start_idx, start_idx + L
        
        # Compute z_threshold
        if flag_end_idx - flag_start_idx > 1000:
            z_threshold = self.config.z_threshold_dict['1000']
        else:
            z_threshold = self.config.z_threshold_dict[str(flag_end_idx - flag_start_idx)]

        # Determine if the z_score indicates a watermark
        is_watermarked = max_z_score > z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            if is_watermarked:
                return {"is_watermarked": is_watermarked, "indices": [(flag_start_idx, flag_end_idx, max_z_score)]}
            else:
                return {"is_watermarked": is_watermarked, "indices": []}
    
    def detect_watermark_with_fix_window(self, text: str, return_dict: bool = True, L: int = 200, threshold: int = 4.0, *args, **kwargs):
        """Detect watermark in the text using fixed window algorithm."""
        
        token_flags = self.get_token_flags(text)

        is_watermarked = False

        indices = []

        # Traverse through the text and calculate the score for each window
        for i in range(self.config.prefix_length, len(token_flags) - L + 1):
            z_score = self.utils.score_sequence_by_window(token_flags, i, i + L)
            if z_score > threshold:
                is_watermarked = True
                indices.append((i, i + L, z_score))

        if return_dict:
            return {"is_watermarked": is_watermarked, "indices": indices}
        else:
            return (is_watermarked, indices)
            
    def detect_watermark_with_seeker(self, text: str, return_dict: bool = True, targeted_fpr=1e-6, window_size=50, threshold_1=0.5, threshold_2=1.5, top_k=20, min_length=100, tolerance=100, *args, **kwargs):
        """Detect watermarked segments in the text using WaterSeeker algorithm."""

        # Get token flags
        token_flags = self.get_token_flags(text)

        # Suspicous segments localization
        indices = self.utils.detect_anomalies(token_flags, window_size, threshold_1, threshold_2, top_k, min_length, tolerance)
        
        if not indices:
            if return_dict:
                return {"is_watermarked": False, "indices": []}
            else:
                return (False, [])
        else:
            is_watermarked = False
            filtered_indices = []

            # Check if the suspicious segments are watermarked
            for indice in indices:
                found_in_current_indice = False  # 标志变量，用于标记当前indice是否找到符合条件的区间

                max_z = -100
                max_index = ()
                
                # Local traverse detection
                for start_idx in range(indice[0], indice[0] + window_size):
                    for end_idx in range(indice[-1], indice[-1] - window_size, -1):
                        if end_idx - start_idx < min_length:
                            break
                        if end_idx - start_idx > 1000:
                            z_threshold = self.config.z_threshold_dict['1000']
                        else:
                            z_threshold = self.config.z_threshold_dict[str(end_idx - start_idx)]
                        z_score = self.utils.score_sequence_by_window(token_flags, start_idx, end_idx)
                        if z_score > z_threshold:
                            is_watermarked = True
                            if z_score > max_z:
                                max_z = z_score
                                max_index = (start_idx, end_idx, z_score)
                            found_in_current_indice = True
                if found_in_current_indice:
                    filtered_indices.append(max_index)
                        
            if return_dict:
                return {"is_watermarked": is_watermarked, "indices": filtered_indices}
            else:
                return (is_watermarked, filtered_indices)

