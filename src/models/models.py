from typing import Dict

from src.models import lcnn, specrnet


def get_model(model_name: str, config: Dict, device: str):
    if model_name == "lcnn":
        return lcnn.LCNN(device=device, **config)
    elif model_name == "specrnet":
        return specrnet.SpecRNet(
            specrnet.get_config(config.get("input_channels", 1)),
            device=device,
            **config,
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported")
