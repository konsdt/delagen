# deep_ela/load_model.py
from glob import glob
from .registry import MODELS
import yaml
import os,  time


from .encoders import EncoderBackbone

class DeepELA(EncoderBackbone):
    def __init__(self, name, path_dir=None, path_ckpt=None, path_cnfg=None, device='cpu'):
        self.name = name
        assert path_dir is not None or (path_ckpt is not None and path_cnfg is not None), \
            'Either path_dir or path_ckpt and path_config must be provided!'
        ## Identify checkpoint and config paths
        if path_dir is not None:
            try:
                path_ckpt = glob(os.path.join(path_dir, '*_backbone.ckpt'))[0]
                path_cnfg = glob(os.path.join(path_dir, 'hparams.yaml'))[0]
            except:
                raise Exception(f'Failed to load model at location {path_dir}!')
        ## Load config and create model
        with open(path_cnfg) as f:
            config = yaml.safe_load(f.read())
        super().__init__(**config)
        ## Load parameters
        import torch
        self.load_state_dict(torch.load(path_ckpt, weights_only=True)) 
        ## Set device and eval
        self.to(device).eval()
        
    def __call__(self, X, y, include_costs=False, repetitions=10):
        start = time.time() # Measure runtime
        features = super().predict(coordinates=X, fvalues=y, repetitions=repetitions, return_embeddings=False)
        features = {f'{self.name}.X{i}': f for i,f in enumerate(features)}
        if include_costs:
            features[f'{self.name}.costs_runtime'] = time.time() - start
        return features

def _cache_dir():
    from torch.hub import get_dir
    from pathlib import Path
    d = Path(get_dir()) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _ensure_hparams(dst, url: str):
    if not dst.exists():
        from torch.hub import download_url_to_file
        download_url_to_file(url, str(dst), progress=True)

def load_deepela(name: str = "medium-50d-v1", device: str = "cpu", strict: bool = True):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")

    urls = MODELS[name]
    cache = _cache_dir()

    # hparams
    hparams_path = cache / f"{name}-hparams.yaml"
    _ensure_hparams(hparams_path, urls["hparams_url"])
    import yaml  # lazy import here
    with open(hparams_path) as f:
        hparams = yaml.safe_load(f.read())

    # ckpt
    from torch.hub import load_state_dict_from_url  # lazy
    state = load_state_dict_from_url(
        urls["ckpt_url"],
        model_dir=str(cache),
        progress=True,
        map_location=device,
    )

    # 3) build model + load weights
    from .inference import DeepELA  # import the heavy module only now
    import torch  # lazy
    model = DeepELA(**hparams)  # or your constructor
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=strict)
    else:
        model.load_state_dict(state, strict=strict)
    model.to(device).eval()
    return model