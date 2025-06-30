import os
import logging

import matplotlib
import torch
import wandb
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def test(cfg: DictConfig):
    # Initialization
    matplotlib.use('Agg')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    from evaluator import DSMEvaluator, print_statistics
    from utils import CheckpointIO, DEFAULT_MODEL_FILE
    from dataset import TomoSARDataset
    from utils import lock_seed
    from generator import DSMGenerator
    from tomosar2height import TomoSAR2Height

    # Clear environment variable for rasterio
    if os.environ.get('PROJ_LIB'):
        del os.environ['PROJ_LIB']

    # Shorthands
    cfg_dataset, cfg_loader, cfg_training, cfg_test, cfg_dsm = (
        cfg['dataset'], cfg['dataloader'], cfg['training'], cfg['test'], cfg['dsm_generation']
    )

    # Output directories
    out_dir_run = os.path.join(cfg_training['out_dir'], f"{cfg_training['run_name']}{cfg.run_suffix}")
    out_dir_tiff = os.path.join(out_dir_run, "tiff_test")
    for d in [out_dir_run, out_dir_tiff]:
        os.makedirs(d, exist_ok=True)

    if cfg_training['lock_seed']:
        lock_seed(0)

    # Disable wandb
    wandb.init(mode='disabled')

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.gpu_id >= 0 else "cpu")
    logging.info(f"Device: {device}")

    # Data loader
    test_dataset = TomoSARDataset('test', cfg_dataset=cfg_dataset, random_sample=False, flip_augm=False,
                                  rotate_augm=False)
    vis_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg_loader['n_workers'], shuffle=False)

    logging.info(f"Dataset path: '{cfg_dataset['path']}'")

    # Model
    model = TomoSAR2Height(cfg).to(device)
    wandb.watch(model)

    # DSM Generator
    generator_dsm = DSMGenerator(
        model=model, device=device, data_loader=vis_loader,
        dsm_pixel_size=cfg_dsm['pixel_size'],
        half_blend_percent=cfg_dsm.get('half_blend_percent', None),
        crs_epsg=cfg_dsm.get('crs_epsg', None),
        use_cloud=cfg.use_cloud, use_image=cfg.use_image, use_footprint=cfg.use_footprint
    )

    evaluator = DSMEvaluator(
        cfg_dataset['dsm_gt_path'], None, cfg_dataset['mask_files']
    )

    # Load checkpoint
    checkpoint_io = CheckpointIO(out_dir_run, model=model, optimizer=None, scheduler=None)
    resume_from = cfg_test.get('check_point', None)
    checkpoint_path = resume_from or os.path.join(out_dir_run, DEFAULT_MODEL_FILE)

    try:
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        load_dict = checkpoint_io.load(checkpoint_path, resume_scheduler=False, device=device)
        logging.info(f"Checkpoint loaded: '{checkpoint_path}'")
    except FileExistsError:
        logging.error("Checkpoint does not exist, cannot proceed with inference.")
        exit()

    n_iter = load_dict.get('n_iter', 0)
    metric_val_best = load_dict.get('loss_val_best', None)
    logging.info(f"Best validation metric: {metric_val_best:.8f}")

    # Inference
    logging.info(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Output path: '{out_dir_run}'")

    def visualize():
        output_path = os.path.join(out_dir_tiff, f"{cfg_training['run_name']}_dsm_{n_iter:06d}.tiff")
        dsm_writer = generator_dsm.generate_dsm(output_path)
        logging.info(f"nDSM saved to '{output_path}'")

        target_dsm = dsm_writer.get_data()
        eval_dict, diff_arr = evaluator.eval(target_dsm, dsm_writer.T)

        eval_path = os.path.join(out_dir_tiff, f"{cfg_training['run_name']}_dsm_{n_iter:06d}_eval.txt")
        print_statistics(eval_dict, f"{cfg_training['run_name']}-iter{n_iter}", 
                        save_to=eval_path, 
                        has_binary=evaluator.has_binary_building,
                        has_ternary=evaluator.has_ternary_building)
        logging.info(f"Evaluation results saved to '{eval_path}'")

        residual_path = os.path.join(out_dir_tiff, f"{cfg_training['run_name']}_residual_{n_iter:06d}.tiff")
        dsm_writer.set_data(diff_arr)
        dsm_writer.write_to_file(residual_path)
        logging.info(f"Residual DSM saved to '{residual_path}'")

    try:
        visualize()
    except IOError as e:
        logging.error(f"Visualization error: {e}")


if __name__ == '__main__':
    test()
