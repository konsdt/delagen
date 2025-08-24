from .registry import MODELS
from .encoders import EncoderBackbone

# create .cache/torch/checkpoints to store downloaded model weights
def _cache_dir():
    # lazy loading
    # FIXME: when does it really make sense?
    from torch.hub import get_dir
    from pathlib import Path
    d = Path(get_dir()) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _ensure_file(dst, url: str):
    if not dst.exists():
        from torch.hub import download_url_to_file
        download_url_to_file(url, str(dst), progress=True)

# wrapper to make trained model callable
class DeepELA(EncoderBackbone):
    def __init__(self, name, path_ckpt=None, path_cnfg=None, device='cpu'):
        self.name = name
        assert (path_ckpt is not None and path_cnfg is not None), \
            'path_ckpt and path_config must be provided!'
        
        ## Load config and create model
        import yaml  # lazy import here
        with open(path_cnfg) as f:
            hparams = yaml.safe_load(f.read())
        super().__init__(**hparams)
        
        ## Load weights and biases
        import torch
        self.load_state_dict(torch.load(path_ckpt, mmap=True, weights_only=True)) 
        
        ## Set device and eval
        self.to(device).eval()
    # FIXME: What about the repetitions parameter? We always predict 10 times?
    def __call__(self, X, y, include_costs=False, repetitions=10):
        if include_costs:
            import time
            start = time.time() # Measure runtime
        features = super().predict(coordinates=X, fvalues=y, repetitions=repetitions, return_embeddings=False)
        features = {f'{self.name}.X{i}': f for i,f in enumerate(features)}
        if include_costs:
            features[f'{self.name}.costs_runtime'] = time.time() - start
        return features



def load_deepela(name: str = "medium-50d-v1", device: str = "cpu", strict: bool = True):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")

    urls = MODELS[name]
    cache = _cache_dir()
    # hparams
    hparams_path = cache / f"{name}-hparams.yaml"
    _ensure_file(hparams_path, urls["hparams_url"])
    # ckpt
    ckpt_path = cache / f"{name}-weights.ckpt"
    _ensure_file(ckpt_path, urls["ckpt_url"])
    
    model = DeepELA(name=name, path_cnfg=hparams_path, path_ckpt=ckpt_path)  # or your constructor
    return model