import gin
import os
import click
import json
import pickle
import torch
import pytorch_lightning as pl
from pathlib import Path

from platune.datasets.dataset import load_data
from platune.model import PLaTune


# BINS_VALUES = 'bins_values.pkl'
# MIN_MAX = 'metadata_attributes.json'  


def search_for_run(run_path, mode="last"):
    if run_path is None: return None
    if ".ckpt" in run_path: return run_path
    ckpts = map(str, Path(run_path).rglob("*.ckpt"))
    ckpts = filter(lambda e: mode in os.path.basename(str(e)), ckpts)
    ckpts = sorted(ckpts)
    if len(ckpts): return ckpts[-1]
    else: return None


@click.command()
@click.option('-d', '--db_path', default="", help='dataset path')
@click.option('-n', '--name', default="", help='Name of the run')
@click.option('-c', '--config', default="v1", help='Name of the gin configuration file to use')
@click.option('-s', '--save_path', default="", help='path to save models checkpoints')
@click.option('--max_steps', default=300_000, help='Maximum number of training steps')
@click.option('--val_every', default=10_000, help='Checkpoint model every n steps')
@click.option('--gpu', default=-1, help='GPU to use')
@click.option('--ckpt', default=None, help='Path to previous checkpoint of the run')
@click.option('--build_cache', is_flag=True, help='wether to load dataset in cache memory for training')
@click.option('--lmdb_keys_filename', default=None, help='lmdb keys filename')
@click.option('--bins_values_file', default=None, help='path to bins_values pkl file to quantize continuous attributes')
@click.option( '--min_max_file', default=None, help='path to bins_values pkl file to quantize continuous attributes')
def main(
        db_path, 
        name, 
        config, 
        save_path, 
        max_steps, 
        val_every,     
        gpu, 
        ckpt, 
        build_cache, 
        lmdb_keys_filename,
        bins_values_file, 
        min_max_file
    ):
    
    # load config
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "platune/configs", f"{config}.gin")
    print('loading config file : ', config_file)
    gin.parse_config_files_and_bindings([config_file], []) 

    # load data
    with gin.unlock_config():
        gin.bind_parameter("dataset.load_data.data_path", db_path)
        if build_cache:
            gin.bind_parameter("dataset.load_data.cache", build_cache)
        if lmdb_keys_filename is not None:
            gin.bind_parameter("dataset.load_data.lmdb_keys_file", lmdb_keys_filename)
    train, val = load_data()

    os.makedirs(os.path.join(save_path, name), exist_ok=True)

    # load min max values / bins continuous descriptors
    continuous_keys = gin.query_parameter('%CONTINUOUS_KEYS')
    min_max_values = []
    bins_values = []
    if len(continuous_keys) > 0:
        if min_max_file is not None and bins_values_file is not None:
            raise ValueError("choose to quantize or not continuous attributes")

        if min_max_file is not None:
            with open(os.path.join(db_path, min_max_file)) as f:
                metadata = json.load(f)

            for k, v in metadata['continuous_attr_min_max'].items():
                if k in continuous_keys:
                    min_max_values.append((v["min"], v["max"]))

        elif bins_values_file is not None:
            with open(os.path.join(db_path, bins_values_file), "rb") as f:
                bins = pickle.load(f)

            for k, v in bins.items():
                if k in continuous_keys:
                    bins_values.append(v)

    # instantiate model
    with gin.unlock_config():
        if len(min_max_values) > 0:
            gin.bind_parameter(
                "model.PLaTune.min_max_attr_continuous", min_max_values)
        if len(bins_values) > 0:
            gin.bind_parameter("model.PLaTune.bins_values", bins_values)
    model = PLaTune()

    # model checkpoints
    callbacks_ckpt = []
    if val is not None:
        validation_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="validation", filename="best")
        callbacks_ckpt.append(validation_checkpoint)
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")
    callbacks_ckpt.append(last_checkpoint)

    val_check = {}
    if val is not None:
        if len(train) >= val_every:
            val_check["val_check_interval"] = val_every
        else:
            nepoch = val_every // len(train)
            val_check["check_val_every_n_epoch"] = nepoch

    # select GPU
    # accelerator = None
    # device = "cuda:" + str(gpu) if torch.cuda.is_available() and gpu >= 0 else "cpu"
    # if device.startswith("cuda"):
    #     accelerator = "cuda"
    accelerator =  "cuda" if torch.cuda.is_available() and gpu >= 0 else None
    if accelerator == "cuda":
        device = gpu
    print(f'device - selected gpu: {accelerator}:{device}')


    # instantiate trainer
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(save_path, name=name),
        accelerator=accelerator,
        devices=device,
        callbacks=callbacks_ckpt,
        max_epochs=100000,
        max_steps=max_steps,
        profiler="simple",
        enable_progress_bar=True,
        **val_check,
    )

    run = search_for_run(ckpt)
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        print("Restarting from step : ", step)
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    with open(os.path.join(os.path.join(save_path, name), "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    # train model
    trainer.fit(model, train, val, ckpt_path=run)


if __name__ == "__main__":
    main()
