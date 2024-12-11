class KaolinStub:
    """Stub implementation for kaolin when GPU is not available."""
    def __getattr__(self, name):
        def method_stub(*args, **kwargs):
            raise RuntimeError(
                f"The method '{name}' cannot be used because kaolin requires an NVIDIA GPU, which is not available."
            )
        return method_stub