"""CT5 pipeline alias.

This module intentionally re-exports the shared prompt decomposition pipeline
implementation to avoid maintaining duplicate code paths.
"""

from .pipeline_prompt_decomposition import (
    PromptDecomposePipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
)

__all__ = ["PromptDecomposePipeline", "rescale_noise_cfg", "retrieve_timesteps"]
