import os
import logging
from datetime import datetime, timedelta

import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import matplotlib
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def train(cfg: DictConfig):
    # Initialization
    matplotlib.use('Agg')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    from evaluator import DSMEvaluator
    from utils import CheckpointIO, DEFAULT_MODEL_FILE
    from dataset import TomoSARDataset
    from utils import lock_seed
    from generator import DSMGenerator
    from trainer import Trainer
    from tomosar2height import TomoSAR2Height

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.gpu_id >= 0 else "cpu")
    logging.info(f"Device: {device}")

    # Environment configuration
    if os.environ.get('PROJ_LIB'):
        del os.environ['PROJ_LIB']

    t_start = datetime.now()
    no_wandb = not cfg.wandb

    # Shorthands
    cfg_training, cfg_model, cfg_dataset, cfg_loader = (
        cfg['training'], cfg['model'], cfg['dataset'], cfg['dataloader']
    )
    batch_size, val_batch_size = cfg_training['batch_size'], cfg_training['val_batch_size']

    # Output directories
    out_dir_run, out_dir_ckpt, out_dir_tiff = (
        os.path.join(cfg_training['out_dir'], f"{cfg_training['run_name']}{cfg.run_suffix}"),
        os.path.join(cfg_training['out_dir'], "check_points"),
        os.path.join(cfg_training['out_dir'], "tiff")
    )
    for d in [out_dir_run, out_dir_ckpt, out_dir_tiff]:
        os.makedirs(d, exist_ok=True)

    if cfg_training['lock_seed']:
        lock_seed(0)

    # Configure wandb
    wandb.init(
        project='tomosar2height',
        config=OmegaConf.to_container(cfg, resolve=True),
        name=os.path.basename(out_dir_run),
        dir=os.path.join(out_dir_run, "wandb"),
        mode='disabled' if no_wandb else 'online',
        settings=wandb.Settings(start_method="fork")
    )

    # Data loaders
    datasets = {
        key: TomoSARDataset(key, cfg_dataset, random_sample=(key == 'train'),
                            random_length=(cfg_training['random_dataset_length'] if key == 'train' else None),
                            flip_augm=cfg_training['augmentation']['flip'] if key == 'train' else False,
                            rotate_augm=cfg_training['augmentation']['rotate'] if key == 'train' else False)
        for key in ['train', 'val', 'vis']
    }
    n_workers = cfg_loader['n_workers']
    torch.set_num_threads(n_workers)

    loaders = {
        key: DataLoader(datasets[key],
                        batch_size=(batch_size if key == 'train' else val_batch_size if key == 'val' else 1),
                        num_workers=n_workers, shuffle=(key == 'train'))
        for key in ['train', 'val', 'vis']
    }

    logging.info(f"Dataset path: '{cfg_dataset['path']}'")
    logging.info(f"Training data: n_data={len(datasets['train'])}, batch_size={batch_size}")
    logging.info(f"Validation data: n_data={len(datasets['val'])}, val_batch_size={val_batch_size}")

    # Model
    model_class = TomoSAR2Height
    model = model_class(cfg).to(device)
    wandb.watch(model)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg_training['learning_rate'])
    schedulers = {
        'CyclicLR': CyclicLR,
        'ReduceLROnPlateau': ReduceLROnPlateau,
        'CosineAnnealingLR': CosineAnnealingLR,
        'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts
    }
    scheduler = schedulers[cfg_training['scheduler']['type']](optimizer, **cfg_training['scheduler']['kwargs'])

    # Trainer and DSM generator
    trainer = Trainer(
        model=model, optimizer=optimizer, device=device,
        optimize_every=cfg_training['optimize_every'],
        use_cloud=cfg.use_cloud, use_image=cfg.use_image,
        use_footprint=cfg.use_footprint, weight_ce=cfg.training.weight_ce
    )

    generator_dsm = DSMGenerator(
        model=model, device=device, data_loader=loaders['vis'],
        dsm_pixel_size=cfg['dsm_generation']['pixel_size'],
        half_blend_percent=cfg['dsm_generation'].get('half_blend_percent', None),
        crs_epsg=cfg['dsm_generation'].get('crs_epsg', None),
        use_cloud=cfg.use_cloud, use_image=cfg.use_image,
        use_footprint=cfg.use_footprint
    )

    evaluator = DSMEvaluator(
        cfg_dataset['dsm_gt_path'], None, {'building': cfg_dataset['mask_files']['building']}
    )

    # Checkpoint handling
    checkpoint_io = CheckpointIO(out_dir_run, model=model, optimizer=optimizer, scheduler=scheduler)
    try:
        load_dict = checkpoint_io.load(cfg_training.get('resume_from', ''),
                                       resume_scheduler=cfg_training.get('resume_scheduler', True),
                                       device=device)
        logging.info('Resuming from previous checkpoint.')
    except FileNotFoundError:
        load_dict = {}
        logging.info('Training from scratch.')
    n_iter, metric_val_best = load_dict.get('n_iter', 0), load_dict.get('loss_val_best', np.inf)

    # Visualize function
    def visualize():
        output_path = os.path.join(out_dir_tiff, f"{cfg_training['run_name']}_dsm_{n_iter:06d}.tiff")
        dsm_writer = generator_dsm.generate_dsm(output_path)
        target_dsm = dsm_writer.get_data()
        eval_dict, diff_arr = evaluator.eval(target_dsm, dsm_writer.T)
        wandb.log({f'nDSM/{k}/{k2}': v2 for k, v in eval_dict.items() for k2, v2 in v.items()}, step=n_iter)

    # Training loop
    while n_iter < cfg.training.max_iteration:
        for batch in loaders['train']:
            if not batch['is_valid'][0]:
                continue

            trainer.train_step(batch)
            if trainer.accumulated_steps == 0:
                n_iter += 1
                training_time = datetime.now() - t_start + timedelta(seconds=load_dict.get('training_time', 0))

                # Log training info
                wandb.log({
                    'iteration': n_iter,
                    'train/loss': trainer.last_avg_loss,
                    'lr': optimizer.param_groups[0]['lr'],
                    'misc/training_time': training_time.total_seconds(),
                    **{f'train/{k}': v for k, v in trainer.last_avg_loss_dict.items()}
                }, step=n_iter)

                if n_iter % cfg_training['print_every'] == 0:
                    logging.info(f"Iteration {n_iter}, Loss = {trainer.last_avg_loss:.5f}")

                if n_iter % cfg_training['checkpoint_every'] == 0:
                    checkpoint_io.save(os.path.join(out_dir_ckpt, DEFAULT_MODEL_FILE),
                                       n_iter=n_iter, loss_val_best=metric_val_best,
                                       training_time=training_time.total_seconds())

                if n_iter % cfg_training['validate_every'] == 0:
                    eval_dict = trainer.evaluate(loaders['val'])
                    metric_val = eval_dict[cfg_training['model_selection_metric']]
                    wandb.log({f"val/{k}": v for k, v in eval_dict.items()}, step=n_iter)
                    if metric_val < metric_val_best:
                        metric_val_best = metric_val
                        checkpoint_io.save(os.path.join(out_dir_ckpt, 'model_best.pt'),
                                           n_iter=n_iter, loss_val_best=metric_val_best,
                                           training_time=training_time.total_seconds())

                if n_iter % cfg_training['visualize_every'] == 0:
                    visualize()

                if cfg_training['scheduler']['type'] in ['CyclicLR', 'CosineAnnealingLR',
                                                         'CosineAnnealingWarmRestarts']:
                    scheduler.step()

    logging.info("Optimization done!")


if __name__ == '__main__':
    train()
