import gin
import copy
import librosa
import lmdb
import torch
import numpy as np
import os
from tqdm import tqdm
import pickle
import pretty_midi
from music2latent import EncoderDecoder
import click

from platune.datasets.parsers import get_parser

from platune.datasets.audio_example import AudioExample

from platune.datasets.transforms import BasicPitchPytorch
from platune.datasets.audio_descriptors import compute_all
from platune.datasets.process_attributes import process_midi_attributes, process_midi_attributesv2, get_midi_notes


torch.set_grad_enabled(False)


def normalize_signal(x: np.ndarray, max_gain_db: int = 30, gain_margin: float = 0.9):
    peak = np.max(abs(x))
    if peak == 0:
        return x
    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)
    return gain_margin * x * gain


def get_midi(midi_data: pretty_midi.PrettyMIDI, chunk_number, num_signal, sample_rate):
    do_silence_check = False

    length = num_signal / sample_rate
    tstart = chunk_number * num_signal / sample_rate
    tend = (chunk_number + 1) * num_signal / sample_rate

    if len(midi_data.instruments) == 0:
        do_silence_check = True
        midi_data = None
        return do_silence_check, midi_data
    
    out_notes = []
    for note in midi_data.instruments[0].notes:
        if note.end > tstart and note.start < tend:
            note.start = max(0, note.start - tstart)
            note.end = min(note.end - tstart, length)
            out_notes.append(note)

    if len(out_notes) == 0:
        do_silence_check = True
        midi_data = None
        return do_silence_check, midi_data

    midi_data.instruments[0].notes = out_notes
    midi_data.adjust_times([0, length], [0, length])

    return do_silence_check, midi_data


@click.command()
@click.option('-i', '--input_path', default="", help='folder with the audio files')
@click.option('-o', '--output_path', default="", help='lmdb save path')
@click.option('-s', '--db_size', default=40, help='Max Size of lmdb database')
@click.option('-p', '--parser_name', default="simple_parser", help='parser function to obtain the list audio files and metadatas')
@click.option('-c', '--config', default=None, help='Name of the gin configuration file to use')
@click.option('-m', '--emb_model_path', default="", help='code to use. Either "music2latent" or a torchscript path')
@click.option('--gpu', default=-1, help='device for basic-pitch and codec (-1 for cpu)')
@click.option('-n', '--num_signal', default=131_072, help='chunk sizes')
@click.option('--sr', default=44_100, help='sample rate')
@click.option('-b', '--batch_size', default=32, help='Batch size (for embedding model inference)')
@click.option('--normalize', is_flag=True, help='Normalize audio waveform (done once per file and not chunks ! )')
@click.option('--cut_silences', is_flag=True, help='Remove silence chunks')
@click.option('--save_waveform', is_flag=True, help='Wether to save the waveform in the lmdb')
@click.option('-l', '--descriptors_list', multiple=True, default=[], help='list of audio descriptors to compute on audio chunks')
@click.option('--use_basic_pitch', is_flag=True, help='use basic pitch for midi extraction from audio')
@click.option('--midi_attributes', is_flag=True, help='Whether to compute and save midi attributes on the midi chunks')
@click.option('-v', '--version', default="v1", help="data processing version")
def main(
        input_path, 
        output_path, 
        db_size, 
        parser_name,
        config,
        emb_model_path, 
        gpu, 
        num_signal, 
        sr, 
        batch_size, 
        normalize,     
        cut_silences, 
        save_waveform, 
        descriptors_list, 
        use_basic_pitch, 
        midi_attributes,
        version
    ):
    
    # cast args to python type
    descriptors_list = list(descriptors_list)

    # load pretrained codec
    device = "cuda:" + str(gpu) if torch.cuda.is_available() and gpu >= 0 else "cpu"

    # logging
    print("-"*60)
    print(" "*20 + "Config:")
    print("-"*60)
    print("Audios path : ", input_path)
    print("Output path : ", output_path)
    print("gin config : ", config)
    print("parser : ", parser_name)
    print("signal length : ", num_signal)
    print("sample rate : ", sr)
    print("pretrained codec : ", emb_model_path)
    print("device : ", device)
    print("-"*60)
    print(" "*20 + "Attributes:")
    print("-"*60)
    print("Audio descriptors to be computed : ", descriptors_list)
    print("Using BasicPitch to extract MIDI data : ", use_basic_pitch)
    print("Processing MIDI attributes (melody, instrument) : ", midi_attributes)
    print("-"*60)


    if emb_model_path == "music2latent":
        emb_model = EncoderDecoder(device=device)
    else:
        emb_model = torch.jit.load(emb_model_path).to(device)
    z_length = None

    # load config for processing attributes
    if config is not None:
        config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "platune/datasets/configs", f"{config}.gin")
        print('loading config file : ', config_file)
        gin.parse_config_files_and_bindings([config_file],[])

        if midi_attributes:
            with gin.unlock_config():
                if version == 'v1':
                    gin.bind_parameter("process_attributes.process_midi_attributes.num_signal", num_signal)
                elif version == 'v2':
                    gin.bind_parameter("process_attributes.process_midi_attributesv2.num_signal", num_signal)
                else:
                    raise ValueError(f'version {version} does not exist!')
                
        if len(descriptors_list) > 0:
            with gin.unlock_config():
                gin.bind_parameter("compute_all.descriptors", descriptors_list)

    # initialize lmdb database
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(
        output_path,
        map_size=db_size * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    # parse audio files
    audio_files, metadatas = get_parser(parser_name)(input_path)

    # loader BasicPitchPytorch
    if use_basic_pitch:
        BP = BasicPitchPytorch(sr=sr, device=device)

    # init
    chunks_buffer, metadatas_buffer, midis = [], [], []
    cur_index = 0
    skip_examples_inf_loud = 0

    # process loop
    for i, (file, metadata) in enumerate(zip(tqdm(audio_files), metadatas)):

        # load audio
        try:
            audio = librosa.load(file, sr=sr)[0]  # only mono not stereo
        except:
            print("error loading file : ", file)
            continue
        audio = audio.squeeze()

        if audio.shape[-1] == 0:
            print("Empty file")
            continue

        if normalize:
            audio = normalize_signal(audio)

        # check audio length to ensure power of 2 (ie. multiple of num signal)
        length = audio.shape[-1]
        if length < num_signal:
            print(f'Warning - skip audio {file} because audio length is too short : {length} < num_signal={num_signal}')
            continue
        else:
            if length % num_signal < num_signal // 2:
                length_crop = (length // num_signal) * num_signal
                audio = audio[:length_crop]
            else:
                # pad audio signal to a power of 2 (num_signal)
                audio = np.pad(audio, (0, num_signal - audio.shape[-1] % num_signal))
        
        # process MIDI data
        if use_basic_pitch:
            midi_data = BP(audio)
        else:
            midi_data = None
        
        # reshape audio signal into chunks
        chunks = audio.reshape(-1, num_signal)
        chunk_index = 0

        # get the number of latent frames computed by the codec
        if i == 0 and emb_model is not None and z_length is None:
            ex_chunk_torch = torch.from_numpy(chunks[0]).to(device)
            if emb_model_path == "music2latent":
                ex_chunk_torch = ex_chunk_torch.reshape(-1, num_signal)
            else:
                ex_chunk_torch = ex_chunk_torch.reshape(-1, 1, num_signal)

            z_ex = emb_model.encode(ex_chunk_torch)
            z_length = z_ex.shape[-1]

            if config is not None:
                if midi_attributes:
                    if version == 'v1':
                        with gin.unlock_config():
                            gin.bind_parameter("process_attributes.process_midi_attributes.z_length", z_length)

                if len(descriptors_list) > 0:
                    with gin.unlock_config():
                        gin.bind_parameter("audio_descriptors.compute_all.z_length", z_length)

        empty_midis_indices = []
        for j, chunk in enumerate(chunks):
            
            # Chunk the midi
            if midi_data is not None:
                silence_test, midi = get_midi(
                    copy.deepcopy(midi_data), 
                    chunk_number=chunk_index,
                    num_signal=num_signal,
                    sample_rate=sr,
                )
                if midi is None:
                    empty_midis_indices.append(j)
            else:
                midi = None
                silence_test = np.max(abs(chunk)) < 0.05 if cut_silences else False

            # don't process buffer if empty slice
            if silence_test:
                chunk_index += 1
                continue

            midis.append(midi)
            chunks_buffer.append(chunk)
            metadatas_buffer.append(metadata)

            if len(chunks_buffer) == batch_size or (j == len(chunks) - 1 and i == len(audio_files) - 1):

                # get latent representation from pretrained codec
                if emb_model is not None:
                    chunks_buffer_torch = torch.from_numpy(np.stack(chunks_buffer)).to(device)
                    if emb_model_path == "music2latent":
                        chunks_buffer_torch = chunks_buffer_torch.reshape(-1, num_signal)
                    else:
                        chunks_buffer_torch = chunks_buffer_torch.reshape(-1, 1, num_signal)
                    z = emb_model.encode(chunks_buffer_torch)
                else:
                    z = [None] * len(chunks_buffer)
                
                for i, (audio_array, z_array, midi, cur_metadata) in enumerate(zip(chunks_buffer, z, midis, metadatas_buffer)):
                    
                    assert audio_array.shape[-1] == num_signal

                    if i in empty_midis_indices:
                        # do not store chunk if you require midi data but midi is None
                        continue

                    # compute audio descriptors
                    feat = compute_all(audio_array) if len(descriptors_list) > 0 else None

                    if 'integrated_loudness' in feat and np.any(feat['integrated_loudness'] == float("-inf")):
                        skip_examples_inf_loud += 1
                        continue

                    if 'loudness1s' in feat and np.any(feat['loudness1s'] == float("-inf")):
                        skip_examples_inf_loud += 1
                        continue
                    
                    # create instance of our lmdb database
                    key = f"{cur_index:08d}"
                    ae = AudioExample()

                    # save chunk audio waveform to lmdb database
                    if save_waveform:
                        if type(audio_array) == torch.Tensor:
                            audio_array = audio_array.cpu().numpy()
                        audio_array = (audio_array * (2**15 - 1)).astype(np.int16)
                        ae.put_array("waveform", audio_array, dtype=np.int16)

                    # save latent representation
                    if z_array is not None:
                        ae.put_array("z", z_array.cpu().numpy(), dtype=np.float32)

                    # save metadata
                    cur_metadata["chunk_index"] = chunk_index
                    cur_metadata["key"] = key
                    ae.put_metadata(cur_metadata)

                    # save MIDI data
                    if midi is not None:
                        ae.put_buffer(key="midi", b=pickle.dumps(midi), shape=None)
                        
                        if midi_attributes:
                            if version == 'v1':
                                attr_midi = process_midi_attributes(x=midi, instrument_val=cur_metadata["instrument"])
                            elif version == 'v2':
                                pitch, onset, offset, _ = get_midi_notes(midi)
                                attr_midi = process_midi_attributesv2(pitch, onset, offset, instrument_val=cur_metadata["instrument"])

                            for k, v in attr_midi.items():
                                ae.put_array(k, v, dtype=np.int32)

                    # save audio descriptors
                    if feat is not None:
                        for k, v in feat.items():
                            ae.put_array(k, v, dtype=np.float32)

                    # save AudioExample instance to lmdb database
                    with env.begin(write=True) as txn:
                        txn.put(key.encode(), bytes(ae))
                    cur_index += 1

                chunks_buffer, midis, metadatas_buffer = [], [], []
            chunk_index += 1

    print("nb of audio chunks skipped because found -inf loudness: ", skip_examples_inf_loud)
    env.close()

    if config is not None:
        # save config if processing attributes
        # NB: done at the end because otherwise the classes/functions that have not been
        # instantiated yet won't appear
        with open(os.path.join(output_path, "config.gin"),"w") as config_out:
            config_out.write(gin.operative_config_str())


if __name__ == '__main__':
    main()
