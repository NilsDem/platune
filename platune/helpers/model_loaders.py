import gin
import pickle
import torch
from typing import Union

from music2latent import EncoderDecoder
from platune.model import PLaTune


def load_model(
        ckpt_path: str,
        config_path: str,
        emb_model_path: str,
        device: Union[str, torch.device] = 'cpu',
        quantized: bool = False,
        bins_values_filepath: str = '',
    ):
    gin.parse_config_file(config_path)
    checkpoint = torch.load(ckpt_path, map_location=device)

    bins_values = []
    if quantized:
        continuous_keys = gin.query_parameter('%CONTINUOUS_KEYS')
        
        with open(bins_values_filepath, "rb") as f:
            bins = pickle.load(f)

        for k, v in bins.items():
            if k in continuous_keys:
                bins_values.append(v)

    if len(bins_values) > 0:
        gin.bind_parameter(
            "tc_plugen.DiffusionPluGeN.bins_values",
            bins_values
        )
    pretrained = PLaTune()

    pretrained.load_state_dict(checkpoint["state_dict"])
    pretrained.eval()
    pretrained = pretrained.to(device)

    if emb_model_path == "music2latent":
        emb_model = EncoderDecoder(device=device)
    else:
        emb_model = torch.jit.load(emb_model_path).to(device)

    return pretrained, emb_model
