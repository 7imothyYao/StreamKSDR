"""Package initializer for mainFunction.

Exposes the primary OnlineKernelSDR class (renamed from RealtimeOnlineKSPCA).
"""

from .OKS_main import OnlineKernelSDR, RealtimeOnlineKSPCA  # Re-export old alias

__all__ = ["OnlineKernelSDR", "RealtimeOnlineKSPCA"]