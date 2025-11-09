#!/usr/bin/env python3
"""Unified wrapper for loading and running transformer models."""

from __future__ import annotations
import logging
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("model_wrapper")


class ModelWrapper:
    """Wrapper for loading and running causal language models."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Load a causal LM and tokenizer.

        Args:
            model_name: HuggingFace model identifier
            device: Optional device override (defaults to cuda if available)
            dtype: Optional dtype override (defaults to float16 on cuda, float32 on cpu)
        """
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        if dtype is None:
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            self.dtype = dtype

        LOGGER.info("Loading model %s on %s with dtype %s", model_name, self.device, self.dtype)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=self.dtype,
        )
        #self.model.to(self.device)
        self.model.eval()

    def tokenize(
        self,
        text: str,
        return_offsets_mapping: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text and move to device.

        Args:
            text: Input text to tokenize
            return_offsets_mapping: Whether to return character offsets

        Returns:
            Dictionary with input_ids, attention_mask, and optionally offset_mapping
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=True,
            return_offsets_mapping=return_offsets_mapping,
        )
        encoded = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        return encoded

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        strip_prompt: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            strip_prompt: Whether to remove prompt from output
            system_prompt: Optional system message (uses chat template if available)
            **kwargs: Additional generation parameters

        Returns:
            Generated text (decoded)
        """
        # Apply chat template if available and system prompt provided
        formatted_prompt = prompt
        if hasattr(self.tokenizer, "apply_chat_template") and system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt},
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        encoded = self.tokenize(formatted_prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
            "no_repeat_ngram_size": kwargs.get("no_repeat_ngram_size", 6),
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        response_ids = generated_ids[0]
        if strip_prompt:
            response_ids = response_ids[input_ids.shape[1]:]

        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def get_hidden_states(
        self,
        text: str,
        target_layers: Iterable[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Capture hidden states at specified layers.

        Args:
            text: Input text to process
            target_layers: Layer indices to capture (0-indexed)

        Returns:
            Dictionary mapping layer_id -> hidden_states tensor (CPU, detached)
        """
        inputs = self.tokenize(text, return_offsets_mapping=True)
        
        # Remove offset_mapping from inputs (not needed for forward pass)
        inputs = {k: v for k, v in inputs.items() if k != "offset_mapping"}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

        hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # type: ignore[attr-defined]
        
        captures: Dict[int, torch.Tensor] = {}
        for layer_id in target_layers:
            # hidden_states[0] is embeddings, layer N is at index N+1
            index = layer_id + 1
            if index >= len(hidden_states):
                raise ValueError(
                    f"Layer {layer_id} exceeds available layers ({len(hidden_states)-1})"
                )
            # Return [seq_len, hidden_dim] by squeezing batch dimension
            captures[layer_id] = hidden_states[index].squeeze(0).detach().cpu()

        return captures

    def token_strings(self, input_ids: torch.Tensor) -> List[str]:
        """
        Convert token IDs to token strings.

        Args:
            input_ids: Tensor of token IDs

        Returns:
            List of token strings
        """
        return self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

    def generate_with_steering(
        self,
        prompt: str,
        steering_vectors: Dict[int, torch.Tensor],
        steering_coefficient: float = 1.0,
        instruction_end_pos: Optional[int] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        strip_prompt: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt with steering vectors injected into the residual stream.

        Args:
            prompt: Input prompt text
            steering_vectors: Dictionary mapping layer_id -> steering_vector tensor [hidden_dim]
            steering_coefficient: Multiplier for steering vectors (default: 1.0)
            instruction_end_pos: Token position where instruction ends (steering applied after this).
                                 If None, steering is applied to all tokens after the prompt.
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            strip_prompt: Whether to remove prompt from output
            system_prompt: Optional system message (uses chat template if available)
            **kwargs: Additional generation parameters

        Returns:
            Generated text (decoded)
        """
        # Apply chat template if available and system prompt provided
        formatted_prompt = prompt
        if hasattr(self.tokenizer, "apply_chat_template") and system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt},
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        encoded = self.tokenize(formatted_prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        # Determine instruction end position
        if instruction_end_pos is None:
            # Default: apply steering to all generated tokens (after prompt)
            instruction_end_pos = input_ids.shape[1]
        
        # Convert steering vectors to tensors on the correct device
        steering_tensors = {}
        for layer_id, vec in steering_vectors.items():
            if isinstance(vec, np.ndarray):
                vec = torch.from_numpy(vec)
            # Ensure vector is on the same device as the model and has correct dtype
            if hasattr(self.model, 'device'):
                # Try to get device from model's first parameter
                device = next(self.model.parameters()).device
            else:
                device = self.device
            vec = vec.to(device).to(self.dtype)
            steering_tensors[layer_id] = vec * steering_coefficient

        # Register forward hooks to inject steering vectors
        hooks = []
        
        def create_steering_hook(layer_id: int, steering_vec: torch.Tensor, start_pos: int):
            """Create a hook that adds steering vector to hidden states after the layer."""
            def hook(module, input_tuple, output):
                # For Llama models, output is a tuple: (hidden_states, past_key_value, ...)
                # We need to modify hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    
                    # Apply steering only to tokens at or after instruction_end_pos
                    if seq_len > start_pos:
                        # Create a new tensor to avoid in-place modification issues
                        modified_hidden = hidden_states.clone()
                        
                        # Expand steering vector to match batch and sequence dimensions
                        # steering_vec: [hidden_dim] -> [1, 1, hidden_dim] then broadcast
                        num_tokens_to_steer = seq_len - start_pos
                        steering_add = steering_vec.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                        steering_add = steering_add.expand(batch_size, num_tokens_to_steer, hidden_dim)
                        
                        # Add steering vector to positions at and after start_pos
                        modified_hidden[:, start_pos:, :] = modified_hidden[:, start_pos:, :] + steering_add
                        
                        # Return modified output tuple
                        return (modified_hidden,) + output[1:]
                    else:
                        # No steering needed if sequence is shorter than start_pos
                        return output
                else:
                    # If output is not a tuple, it's just hidden_states
                    hidden_states = output
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    
                    if seq_len > start_pos:
                        modified_hidden = hidden_states.clone()
                        num_tokens_to_steer = seq_len - start_pos
                        steering_add = steering_vec.unsqueeze(0).unsqueeze(0)
                        steering_add = steering_add.expand(batch_size, num_tokens_to_steer, hidden_dim)
                        modified_hidden[:, start_pos:, :] = modified_hidden[:, start_pos:, :] + steering_add
                        return modified_hidden
                    else:
                        return output
            
            return hook

        # Register hooks for each steering layer
        try:
            # For Llama models, layers are at model.model.layers[layer_id]
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layers_module = self.model.model.layers
                for layer_id, steering_vec in steering_tensors.items():
                    if layer_id < len(layers_module):
                        hook = layers_module[layer_id].register_forward_hook(
                            create_steering_hook(layer_id, steering_vec, instruction_end_pos)
                        )
                        hooks.append(hook)
                    else:
                        LOGGER.warning(f"Layer {layer_id} not found in model (max: {len(layers_module)-1})")
            else:
                LOGGER.error("Model structure not recognized for steering injection")
                raise ValueError("Cannot inject steering vectors: model structure not supported")
        except Exception as e:
            LOGGER.error(f"Error registering steering hooks: {e}")
            # Clean up hooks if registration fails
            for hook in hooks:
                hook.remove()
            raise

        try:
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
                "no_repeat_ngram_size": kwargs.get("no_repeat_ngram_size", 6),
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

        response_ids = generated_ids[0]
        if strip_prompt:
            response_ids = response_ids[input_ids.shape[1]:]

        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
