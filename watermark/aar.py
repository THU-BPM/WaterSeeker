# ============================================
# aar.py
# Description: Implementation of Aar algorithm
# ============================================

import json
import scipy
import torch
import numpy as np
from math import log
from transformers import LogitsProcessor
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig

class AARConfig:
    """Config class for AAR algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the AAR configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/AAR.json')
        else:
            config_dict = load_config_file(algorithm_config)

        self.prefix_length = config_dict['prefix_length']
        self.eps = config_dict['eps']
        self.threshold = config_dict['threshold']
        self.sequence_length = config_dict['sequence_length']
        self.seed = config_dict['seed']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
        self.temperature = self.gen_kwargs.get('temperature', config_dict['temperature'])


class AARUtils:
    """Utility class for AAR algorithm, contains helper functions."""

    def __init__(self, config: AARConfig, *args, **kwargs) -> None:
        """
            Initialize the AAR utility class.

            Parameters:
                config (AARConfig): Configuration for the AAR algorithm.
        """
        self.config = config
        self.generator = torch.Generator().manual_seed(self.config.seed)
        self.uniform = torch.clamp(torch.rand((self.config.vocab_size * self.config.prefix_length, self.config.vocab_size), 
                                         generator=self.generator, dtype=torch.float32), min=self.config.eps)
        self.gumbel = (-torch.log(torch.clamp(-torch.log(self.uniform), min=self.config.eps))).to(self.config.device)
    
    def detect_anomalies(self, token_scores, window_size=50, threshold_1=0.5, threshold_2=1.5, top_k=20, min_length=100, tolerance=100):
        """Detect anomalies in the token scores."""

        # Calculate the moving average of the token scores
        proportions = []
        for i in range(len(token_scores) - window_size + 1):
            window = token_scores[i:i + window_size]
            proportion = sum(window) / window_size
            proportions.append(proportion)

        # Calculate the mean and standard deviation of the proportions
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

        # Return the valid segments
        if valid_segments:
            return valid_segments
        else:
            return None

class AARLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for AAR algorithm, process logits to add watermark."""

    def __init__(self, config: AARConfig, utils: AARUtils, *args, **kwargs) -> None:
        """
            Initialize the AAR logits processor.

            Parameters:
                config (AARConfig): Configuration for the KGW algorithm.
                utils (AARUtils): Utility class for the KGW algorithm.
        """
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        prev_token = torch.sum(input_ids[:, -self.config.prefix_length:], dim=-1)  # (batch_size,)
        gumbel = self.utils.gumbel[prev_token]  # (batch_size, vocab_size)
        return scores[..., :gumbel.shape[-1]] / self.config.temperature + gumbel


class AAR:
    """Top-level class for the AAR algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the AAR algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = AARConfig(algorithm_config, transformers_config)
        self.utils = AARUtils(self.config)
        self.logits_processor = AARLogitsProcessor(self.config, self.utils)
    
    def watermark_logits_argmax(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.LongTensor:
        """
        Applies watermarking to the last token's logits and returns the argmax for that token.
        Returns tensor of shape (batch,), where each element is the index of the selected token.
        """
        
        # Get the logits for the last token
        last_logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # Get the argmax of the logits
        last_token = torch.argmax(last_logits, dim=-1).unsqueeze(-1)  # (batch,)
        return last_token

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the AAR algorithm."""

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)

        # Initialize
        inputs = encoded_prompt
        attn = torch.ones_like(encoded_prompt)
        past = None

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                    output_gumbel = self.logits_processor(input_ids=inputs, scores=output.logits)
                else:
                    output = self.config.generation_model(inputs)
                    output_gumbel = self.logits_processor(input_ids=inputs, scores=output.logits)

            # Sample token to add watermark
            token = self.watermark_logits_argmax(inputs, output_gumbel)

            # Update past
            past = output.past_key_values

            # Update inputs
            inputs = torch.cat([inputs, token], dim=-1)

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

        return watermarked_text

    def get_token_score(self, text: str):
        """Get the token score for the text."""
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]

        seq_len = len(encoded_text)
        score = [0 for _ in range(seq_len)]

        for i in range(self.config.prefix_length, seq_len):
            prev_tokens_sum = torch.sum(encoded_text[i - self.config.prefix_length:i], dim=-1)
            token = encoded_text[i]
            u = self.utils.uniform[prev_tokens_sum, token]
            score[i] = log(1 / (1 - u))
        
        return score

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Full-text Detection."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]

        seq_len = len(encoded_text)
        score = 0
        for i in range(self.config.prefix_length, seq_len):
            prev_tokens_sum = torch.sum(encoded_text[i - self.config.prefix_length:i], dim=-1)
            token = encoded_text[i]
            u = self.utils.uniform[prev_tokens_sum, token]
            score += log(1 / (1 - u))
        
        p_value = scipy.stats.gamma.sf(score, seq_len - self.config.prefix_length, loc=0, scale=1)
        
        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = bool(p_value < self.config.threshold)

        # Return results based on the `return_dict` flag
        if return_dict:
            if is_watermarked:
                return {"is_watermarked": is_watermarked, "indices": [(0, len(encoded_text), p_value)]}
            else:
                return {"is_watermarked": is_watermarked, "indices": []}
    
    def detect_watermark_win_max(self, text:str, return_dict: bool = True, min_L: int = 1, max_L=None, window_interval=1, *args, **kwargs):
        """Detect watermarked segments in the text using WinMax algorithm."""
        
        # Get token score
        token_scores = self.get_token_score(text)

        if max_L is None:
            max_L = len(token_scores) - self.config.prefix_length

        min_p_value = float('inf')
        flag_start_idx, flag_end_idx = -1, -1

        # Traverse all possible segments
        for L in range(min_L, max_L + 1, window_interval):
            for start_idx in range(self.config.prefix_length, len(token_scores) - L + 1):
                p_value = scipy.stats.gamma.sf(sum(token_scores[start_idx:start_idx + L]), L, loc=0, scale=1)
                if p_value < min_p_value:
                    min_p_value = p_value
                    flag_start_idx, flag_end_idx = start_idx, start_idx + L

        # Determine if the z_score indicates a watermark
        is_watermarked = bool(min_p_value < self.config.threshold)

        # Return results based on the return_dict flag
        if return_dict:
            if is_watermarked:
                return {"is_watermarked": is_watermarked, "indices": [(flag_start_idx, flag_end_idx, min_p_value)]}
            else:
                return {"is_watermarked": is_watermarked, "indices": []}
    
    def detect_watermark_with_fix_window(self, text: str, L: int, threshold: float, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text using fixed window algorithm."""
        
        # Get token score
        token_scores = self.get_token_score(text)

        is_watermarked = False

        indices = []

        # Traverse through the text and calculate the score for each window
        for i in range(self.config.prefix_length, len(token_scores) - L + 1):
            score = sum(token_scores[i:i + L])
            score = scipy.stats.gamma.sf(score, L, loc=0, scale=1)
            if score < threshold:
                is_watermarked = True
                indices.append((i, i + L, score))

        if return_dict:
            return {"is_watermarked": is_watermarked, "indices": indices}
        else:
            return (is_watermarked, indices)
            
    def detect_watermark_with_seeker(self, text: str, return_dict: bool = True, targeted_fpr=1e-4, window_size=50, threshold_1=0.5, threshold_2=1.5, top_k=20, min_length=50, tolerance=100, *args, **kwargs):
        """Detect watermarked segments in the text using WaterSeeker algorithm."""

        # get token score
        token_scores = self.get_token_score(text)

        # Suspicous segments localization
        indices = self.utils.detect_anomalies(token_scores, window_size, threshold_1, threshold_2, top_k, min_length, tolerance)

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
                found_in_current_indice = False
                
                min_p = float('inf')
                min_index = ()

                # Local traverse detection
                for start_idx in range(indice[0], indice[0] + window_size):
                    for end_idx in range(indice[-1], indice[-1] - window_size, -1):
                        if end_idx - start_idx < min_length:
                            break
                        p_value = scipy.stats.gamma.sf(sum(token_scores[start_idx:end_idx]), end_idx - start_idx, loc=0, scale=1)
                        if p_value < targeted_fpr:
                            is_watermarked = True
                            if p_value < min_p:
                                min_p = p_value
                                min_index = (start_idx, end_idx, p_value)
                            found_in_current_indice = True  
                if found_in_current_indice:
                    filtered_indices.append(min_index)

            if return_dict:
                return {"is_watermarked": is_watermarked, "indices": filtered_indices}
            else:
                return (is_watermarked, filtered_indices)
    
    def detect_watermark_with_seeker_phase_1(self, text: str, return_dict: bool = True, targeted_fpr=1e-4, window_size=50, threshold_1=0.5, threshold_2=1.5, top_k=20, min_length=50, tolerance=100, *args, **kwargs):
        """Detect watermarked segments in the text using WaterSeeker algorithm."""

        # get token score
        token_scores = self.get_token_score(text)

        # Suspicous segments localization
        indices = self.utils.detect_anomalies(token_scores, window_size, threshold_1, threshold_2, top_k, min_length, tolerance)

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
                p_value = scipy.stats.gamma.sf(sum(token_scores[indice[0]:indice[-1]]), indice[-1] - indice[0], loc=0, scale=1)
                if p_value < targeted_fpr:
                    is_watermarked = True
                    filtered_indices.append((indice[0], indice[-1], p_value))

            if return_dict:
                return {"is_watermarked": is_watermarked, "indices": filtered_indices}
            else:
                return (is_watermarked, filtered_indices)
