import os
import pathlib
import numpy as np
import torch
import torchaudio
from typing import Dict

from .basic_pitch_torch.model import BasicPitchTorch
from .basic_pitch_torch.inference import predict


class BaseTransform():

    def __init__(self, sr, name) -> None:
        self.sr = sr
        self.name = name

    def forward(self, x: np.array) -> Dict[str, np.array]:
        return None


class BasicPitchPytorch(BaseTransform):

    def __init__(self, sr, device="cpu") -> None:
        super().__init__(sr, "basic_pitch")

        self.pt_model = BasicPitchTorch()

        file_path = pathlib.Path(__file__).parent.resolve()

        self.pt_model.load_state_dict(
            torch.load(
                os.path.join(
                    file_path,
                    'basic_pitch_torch/assets/basic_pitch_pytorch_icassp_2022.pth'
                )))
        self.pt_model.eval()
        self.pt_model.to(device)
        self.device = device

    @torch.no_grad
    def __call__(self, waveform, **kwargs):
        if type(waveform) != torch.Tensor:
            waveform = torch.from_numpy(waveform).to(self.device)

        if self.sr != 22050:
            waveform = torchaudio.functional.resample(waveform=waveform,
                                                      orig_freq=self.sr,
                                                      new_freq=22050)

        #print(waveform)
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            results = []
            for wave in waveform:
                _, midi_data, _ = predict(model=self.pt_model,
                                          audio=wave.squeeze().cpu(),
                                          device=self.device)
                results.append(midi_data)
            return results
        else:
            _, midi_data, _ = predict(model=self.pt_model,
                                      audio=waveform.squeeze().cpu(),
                                      device=self.device)
            return midi_data
