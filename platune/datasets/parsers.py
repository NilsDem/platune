import os
import pathlib
import csv
import yaml
import random
from tqdm import tqdm

from typing import Iterable, Sequence


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm


def search_for_audios(
    path_list: Sequence[str],
    extensions: Sequence[str] = [
        "wav", "opus", "mp3", "aac", "flac", "aif", "ogg"
    ],
):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f"*.{ext}"))
    audios = flatten(audios)
    audios = [str(a) for a in audios if 'MACOS' not in str(a)]

    return audios


def simple_parser(audio_folder, filters=None):
    audios = search_for_audios([audio_folder])
    audios = map(str, audios)
    audios = map(os.path.abspath, audios)
    audios = [*audios]

    if filters is not None:
        audios = [
            a for a in audios if any([s.lower() in a.lower() for s in filters])
        ]

    random.shuffle(audios)

    metadatas = [{"path": audio} for audio in audios]
    print(len(audios), " files found")

    return audios, metadatas


def maestro_parser(main_folder, filters=None):

    # audios = os.listdir(os.path.join(main_folder, "audio"))
    audios = search_for_audios([main_folder])
    metadatas = []
    for a in audios:
        midi_file = a[:-4] + ".midi"
        # audio_file = os.path.join(main_folder, "audio", a)
        metadatas.append({"path": str(a), "midi_file": midi_file})

    # audios = [os.path.join(main_folder, "audio", f) for f in audios]

    print("sanity check: ", audios[0], metadatas[0])

    return audios, metadatas


def synthetic_parser(main_folder, filters=None):

    audios = os.listdir(os.path.join(main_folder, "audio"))
    metadatas = []
    for a in audios:
        midi_file = os.path.join(main_folder, "midi", a.split("_")[0]) + ".mid"
        instrument = "_".join(a.split("_")[1:])[:-4]
        metadatas.append({"midi_file": midi_file, "instrument": instrument})

    audios = [os.path.join(main_folder, "audio", f) for f in audios]

    print("sanity check: ", audios[0], metadatas[0])

    return audios, metadatas


def slakh_parser(audio_folder, ban_list=[]):
    tracks = [
        os.path.join(audio_folder, subfolder)
        for subfolder in os.listdir(audio_folder)
    ]
    meta = tracks[0] + "/metadata.yaml"
    ban_list = [
        "Chromatic Percussion",
        "Drums",
        "Percussive",
        "Sound Effects",
        "Sound effects",
    ]  # , "Ethnic", "Organ", "Synth Pad", "Synth Lead", "Reed"

    #get_list = ["Strings", "Strings (continued)"]
    instr = []
    stem_list = []
    metadata = []
    total_stems = 0

    for trackfolder in tqdm(tracks):
        meta = trackfolder + "/metadata.yaml"
        with open(meta, "r") as file:
            d = yaml.safe_load(file)
        for k, stem in d["stems"].items():
            if stem["inst_class"] not in ban_list:
                stem_list.append(trackfolder + "/stems/" + k + ".flac")
                instr.append(stem["inst_class"])
                metadata.append(stem)
            total_stems += 1

    print(set(instr), "instruments remaining")
    print(total_stems, "stems in total")
    print(len(stem_list), "stems retained")
    audios = stem_list
    metadatas = [{
        "path": audio,
        "instrument": inst
    } for audio, inst in zip(audios, instr)]
    return audios, metadatas


def get_urmp_midi_file_path(midi_folder, audio_path):
    _, n, inst, audio_idx, audio_name = audio_path.stem.split('_')

    midi_files_candidates = list(
        pathlib.Path(midi_folder).rglob(
            f'{audio_idx}_{audio_name}_{inst}*.mid'))

    if len(midi_files_candidates) > 1:

        if (inst == 'vn' and n == '1') or (inst == 'fl' and n == '1') or (
                inst == 'tpt'
                and n == '1') or (inst == 'sax'
                                  and n == '1') or (inst == 'va' and n == '3'):
            midi_filepath = [
                str(p) for p in midi_files_candidates
                if f"{inst}.mid" in str(p)
            ][0]

        elif (inst == 'vn' and n == '2') or (inst == 'fl' and n == '2') or (
                inst == 'tpt'
                and n == '2') or (inst == 'sax'
                                  and n == '2') or (inst == 'va' and n == '4'):
            midi_filepath = [
                str(p) for p in midi_files_candidates if f"{inst}_" in str(p)
            ][0]

        else:
            raise ValueError(
                f'Could not find midi file for file : {str(audio_path)}')

    else:
        midi_filepath = str(midi_files_candidates[0])
    return midi_filepath


def urmp_parser(audio_folder):
    audios = search_for_audios([audio_folder])

    instruments = {
        'vn': 'Violin',
        'va': 'Viola',
        'vc': 'Cello',
        'db': 'Double Basse',
        'fl': 'Flute',
        'ob': 'Oboe',
        'cl': 'Clarinet',
        'sax': 'Saxophone',
        'bn': 'Bassoon',
        'tpt': 'Trumpet',
        'hn': 'Horn',
        'tbn': 'Trombone',
        'tba': 'Tuba',
    }

    metadata = []
    for audio in tqdm(audios):
        inst = pathlib.Path(audio).stem.split("_")[2]
        inst_name = instruments[inst]
        data = {
            "path": audio,
            "instrument": inst_name,
        }
        metadata.append(data)

    print(len(audios), " files found")
    print("sanity check: ", audios[0], metadata[0])

    return audios, metadata


def medley_solos_mono_parser(audio_folder):
    audios = search_for_audios([audio_folder])

    metadata_filepath = os.path.join(
        str(pathlib.Path(audios[0]).parent.parent),
        "Medley-solos-DB_metadata.csv")

    # all_inst_list = [
    #     'clarinet',
    #     'distorted electric guitar',
    #     'female singer',
    #     'flute',
    #     'piano',
    #     'tenor saxophone',
    #     'trumpet',
    #     'violin'
    # ]

    filter_ployphonic_instruments = ['distorted electric guitar', 'piano']

    raw_metadata = {}
    with open(metadata_filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_metadata[row['uuid4']] = {'instrument': row['instrument']}

    metadata = []
    audios_filtered = []

    for audio in tqdm(audios):

        uuid = pathlib.Path(audio).stem.split("_")[-1]

        inst_name = raw_metadata[uuid]['instrument']

        if inst_name not in filter_ployphonic_instruments:
            audios_filtered.append(audio)

            if inst_name == "female singer":
                inst_name = "voice"
            elif inst_name == "tenor saxophone":
                inst_name = "saxophone"
            elif inst_name == "violin":
                inst_name = "strings"

            data = {
                "path": audio,
                "instrument": inst_name,
            }
            metadata.append(data)

    print(len(audios), " files found")
    print(f"selected files : {len(audios_filtered)} / {len(audios)}")
    print("sanity check: ", audios_filtered[0], metadata[0])

    return audios_filtered, metadata


def get_parser(parser_name):
    if parser_name == "simple_parser":
        parser = simple_parser
    elif parser_name == "synthetic_parser":
        parser = synthetic_parser
    elif parser_name == "urmp_parser":
        parser = urmp_parser
    elif parser_name == "medley_solos_mono_parser":
        parser = medley_solos_mono_parser
    elif parser_name == "maestro_parser":
        parser = maestro_parser
    else:
        raise NotImplementedError(f'No parser method named : {parser_name}.')
    return parser
