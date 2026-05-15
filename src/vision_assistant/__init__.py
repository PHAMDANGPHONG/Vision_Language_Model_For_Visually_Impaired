"""Vision Assistant — Multi-layer on-device VLM for the visually impaired.

Architecture (see README §5):
    Layer 1: layer1_input  — Input Filter (blur, SSIM, exposure)
    Layer 2: layer2_vlm    — VLM Core + Confidence Verifier + Re-acquisition
    Layer 3: layer3_output — Output Composer (Spatial Grounding, Priority, VI formatter)
"""

__version__ = "0.1.0"
__author__ = "Thesis Author"
