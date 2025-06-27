# PLaTune: Pretrained Latents Tuner - Adding temporal musical controls on top of pretrained generative models

This repository is linked to our paper submission to the [ISMIR 25](https://ismir2025.ismir.net/) Conference. 

## Abstract 
Recent advances in deep generative modeling have enabled      high-quality models for musical audio synthesis. However, these approaches remain difficult to control, confined to simple, static attributes and, most importantly, entail retraining a different computationally-heavy architecture for each new control. This is inefficient and impractical as it requires substantial computational resources.\
In this paper, we propose a novel approach allowing to add time-varying musical controls on top of any pretrained generative models with an exposed latent space (e.g. neural audio codecs), without retraining or finetuning. Our method supports both discrete and continuous attributes by adapting a rectified flow approach with a latent diffusion transformer. We learn an invertible mapping between pretrained latent variables and a new space disentangling explicit control attributes and *style* variables that capture the remaining factors of variation.\
This enables both feature extraction from an input, but also editing those features to generate transformed audio samples. Finally, this also introduces the ability to perform synthesis directly from the audio descriptors. We validate our method with 4 datasets going from different musical instruments up to full music recordings, on which we outperform state-of-the-art task-specific baselines in terms of both generation quality and accuracy of the control by inferring transferred attributes.


# Install

Create a virtual environment with Python>=3.12 and install dependencies:
```bash
$ pip install -r requirements.txt
```

For training on GPUs, install the Pytorch libraries compatible with your CUDA installation:

```bash
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

*Note: If you encounter some intall errors please check this [section](#fixing-some-install-errors).*


# Data preparation

Process your audio data into a LMDB database and precompute both the latent representations of the pretrained neural audio codec you want to control and the control attributes that we will define your control space.

To do so, use the `prepare_dataset.py` script. The specifications are:
```bash
Usage: prepare_dataset.py [OPTIONS]

Options:
  -i, --input_path TEXT        folder with the audio files
  -o, --output_path TEXT       lmdb save path
  -s, --db_size INTEGER        Max Size of lmdb database
  -p, --parser_name TEXT       parser function to obtain the list audio files
                               and metadatas
  -c, --config TEXT            Name of the gin configuration file to use
  -m, --emb_model_path TEXT    code to use. Either "music2latent" or a
                               torchscript path
  --gpu INTEGER                device for basic-pitch and codec (-1 for cpu)
  -n, --num_signal INTEGER     chunk sizes
  --sr INTEGER                 sample rate
  -b, --batch_size INTEGER     Batch size (for embedding model inference)
  --normalize                  Normalize audio waveform (done once per file
                               and not chunks ! )
  --cut_silences               Remove silence chunks
  --save_waveform              Wether to save the waveform in the lmdb
  -l, --descriptors_list TEXT  list of audio descriptors to compute on audio
                               chunks
  --use_basic_pitch            use basic pitch for midi extraction from audio
  --midi_attributes            Whether to compute and save midi attributes on
                               the midi chunks
  --help                       Show this message and exit.
```

For instance, to prepare MedleySolos data using Music2Latent official pretrained codec with controls on melody, instruments, and basic audio descriptors, you can do:
```bash
(myenv)$ python scripts/prepare_dataset.py -i /path/to/audios -o /path/to/processed_m2l -s 64 -p medley_solos_mono_parser -c medley_solos -m music2latent --gpu 1 -n 131072 --sr 44100 -b 32 --cut_silences --save_waveform -l rms -l centroid -l bandwidth -l booming -l sharpness -l integrated_loudness -l loudness1s --use_basic_pitch --midi_attributes
```

# Train

```bash
```

# Inference

```bash
```

---
---
## Fixing some install errors

- If you are using music2latent pretrained codec with Python>=3.12, you may encounter the following error:
```bash
 File "/path/to/env/lib/python3.12/site-packages/torch/serialization.py", line 1524, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 

        (1) In PyTorch 2.6, we changed the default value of the weights_only argument in torch.load from False to True. Re-running torch.load with weights_only set to False will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with weights_only=True please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default. Please use torch.serialization.add_safe_globals([numpy.core.multiarray.scalar]) or the torch.serialization.safe_globals([numpy.core.multiarray.scalar]) context manager to allowlist this global if you trust this class/function.
```

A fix quick is to specify `weights_only=False` in  `/path/to/env/lib/python3.12/site-packages/music2latent/inference.py`:
```python
class EncoderDecoder:
    ...
    def get_models(self):
        gen = UNet().to(self.device)
        checkpoint = torch.load(self.load_path_inference, map_location=self.device, weights_only=False)
        gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
        self.gen = gen
    ...
```

# Citation
```
@inproceedings{nabi2025platune,
  title={Adding temporal musical controls on top of pretrained generative models},
  author={Nabi, Sarah and Demerl{\'e}, Nils and Peeters, Geoffroy and Bevilacqua, Fr{\'e}d{\'e}ric and Esling, Philippe},
  booktitle={International Society for Music Information Retrieval, ISMIR 2025},
  year={2025}
}
```
