#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Train the 2D Linear Amplitude Encoder (LAE) on DIV2K, against the amplitude-discard baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from instantrph.common import DATASET_2D_DIR, OUTPUT_DIR, PHASE_FILE_2D, pick_device, seed_all
from instantrph.constants import C, R
from instantrph.data_2d import build_loaders
from instantrph.hologram_2d import Hologram2D
from instantrph.metrics import psnr
from instantrph.models import IdentityEncoder, LinearAmplitudeEncoder
from instantrph.viz import hologram_mosaic, lae_output_mosaic

# Fixed ranges for the LAE scale/bias snapshot PNGs (scale starts at 1, bias at 0), so every
# snapshot of a run shares one color scale and can be stored as 8-bit PNG.
LAE_SCALE_RANGE = (-0.5, 2.5)
LAE_BIAS_RANGE = (-1.5, 3.0)


@torch.no_grad()
def evaluate(hologram: Hologram2D, model: nn.Module, loader, device: torch.device) -> tuple[float, float, float, float]:
    """Mean (loss, PSNR) of the LAE and of the amplitude-discard baseline: (loss, psnr, baseline_loss, baseline_psnr)."""
    model.eval()
    baseline = IdentityEncoder(hologram.tmx)
    totals = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for images in loader:
        images = images.to(device)
        preds, preds_baseline = hologram(model, images), hologram(baseline, images)
        batch_size = images.shape[0]
        for i, value in enumerate((
            F.l1_loss(preds, images), psnr(preds, images).mean(),
            F.l1_loss(preds_baseline, images), psnr(preds_baseline, images).mean(),
        )):
            totals[i] += value.item() * batch_size
        count += batch_size
    loss, val_psnr, baseline_loss, baseline_psnr = (t / max(count, 1) for t in totals)
    return loss, val_psnr, baseline_loss, baseline_psnr


@torch.no_grad()
def tensorboard_mosaics(hologram: Hologram2D, model: nn.Module, loader, device: torch.device, args: argparse.Namespace):
    """Reconstruction and LAE-output mosaics for the first batch of `loader`."""
    model.eval()
    images = next(iter(loader)).to(device)
    raw_outputs = model(images.sqrt())
    preds_baseline = hologram(IdentityEncoder(hologram.tmx), images)
    preds = hologram(model, images)
    size = (args.tb_preview_rows, args.tb_preview_cols)
    reconstructions = hologram_mosaic(
        images, preds_baseline, preds, psnr(preds_baseline, images), psnr(preds, images),
        distance_mm=hologram.distance_m * 1000.0, max_images=args.tb_max_images, size=size,
    )
    return reconstructions.cpu(), lae_output_mosaic(images.sqrt(), raw_outputs, max_images=args.tb_max_images, size=size).cpu()


def _save_normalized_png(chw: torch.Tensor, vmin: float, vmax: float, path: Path) -> None:
    norm = ((chw - vmin) / max(vmax - vmin, 1e-12)).clamp(0.0, 1.0)
    Image.fromarray((norm.permute(1, 2, 0).numpy() * 255.0).astype('uint8'), mode='RGB').save(path)


@torch.no_grad()
def save_snapshot(
    model: LinearAmplitudeEncoder,
    hologram: Hologram2D,
    val_loader,
    device: torch.device,
    viz_dir: Path,
    snap_idx: int,
    epoch: int,
    global_step: int,
    max_images: int,
    image_indices: list[int],
) -> None:
    """Save the LAE maps (frame-averaged and per frame), a few reconstructions and the
    validation PSNR, for analysis/visualize_lae_evolution.py."""
    was_training = model.training
    _, val_psnr, _, val_baseline_psnr = evaluate(hologram, model, val_loader, device)
    snap_dir = viz_dir / f'snap_{snap_idx:05d}'
    snap_dir.mkdir(parents=True, exist_ok=True)

    channels = 3
    for name, param, (vmin, vmax) in (('lae_scale', model.scale, LAE_SCALE_RANGE), ('lae_bias', model.bias, LAE_BIAS_RANGE)):
        maps = param.detach().float().cpu().squeeze(0)
        maps = maps.view(channels, maps.shape[0] // channels, *maps.shape[-2:])  # [C, tmx, H, W]
        _save_normalized_png(maps.mean(dim=1), vmin, vmax, snap_dir / f'{name}.png')
        if maps.shape[1] > 1:
            for frame in range(maps.shape[1]):
                _save_normalized_png(maps[:, frame], vmin, vmax, snap_dir / f'{name}_repeat{frame:02d}.png')

    # Reconstructions of the requested validation indices (or the first `max_images`).
    wanted = set(image_indices) if image_indices else set(range(max_images))
    hologram_psnr: list[float] = []
    index = 0
    for images in val_loader:
        if len(hologram_psnr) == len(wanted):
            break
        images = images.to(device)
        preds = hologram(model, images).clamp(0.0, 1.0).float()
        batch_psnr = psnr(preds, images).cpu()
        for pred, value in zip(preds.cpu(), batch_psnr):
            if index in wanted and len(hologram_psnr) < len(wanted):
                arr = (pred.permute(1, 2, 0).numpy() * 255.0).astype('uint8')
                Image.fromarray(arr, mode='RGB').save(snap_dir / f'hologram_{len(hologram_psnr):02d}.png')
                hologram_psnr.append(value.item())
            index += 1
    if image_indices and len(hologram_psnr) < len(wanted):
        print(f'[Viz] Requested {len(wanted)} val indices but only found {len(hologram_psnr)} in loader.')

    (snap_dir / 'meta.json').write_text(json.dumps({'epoch': epoch, 'global_step': global_step, 'hologram_psnr': hologram_psnr}))
    csv_path = viz_dir / 'psnr_history.csv'
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['snap_idx', 'epoch', 'global_step', 'val_psnr', 'val_dummy_psnr'])
        writer.writerow([snap_idx, epoch, global_step, val_psnr, val_baseline_psnr])
    if was_training:
        model.train()


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = pick_device(args.device)
    image_hw = (args.image_rows, args.image_cols)
    print(f'[Init] Device: {device}')
    print(f'[Init] Image size (R, C): {image_hw}')

    train_loader, val_loader = build_loaders(
        args.datasets_root, args.train_subdir, args.val_subdir, image_hw=image_hw, batch_size=args.batch_size,
        num_workers=args.num_workers, debug_sample_index=args.debug_sample_index if args.debug_single_sample else None,
    )
    print(f'[Data] Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}')
    print(f'[Data] Train batches/epoch: {len(train_loader)} | Batch size: {args.batch_size}')
    viz_image_indices = sorted({int(i) for i in args.viz_image_indices.split(',') if i.strip()})

    model = LinearAmplitudeEncoder(tmx=args.tmx, image_hw=image_hw).to(device)
    print(f'[Model] Trainable parameters: {sum(p.numel() for p in model.parameters()):,}')
    hologram = Hologram2D.from_phase_file(
        args.phase_file, tmx=args.tmx, layer_id=args.layer_id, distance_m=args.target_distance_mm / 1000.0,
        image_hw=image_hw, device=device,
    )
    print(f'[Optics] Phase pattern {tuple(hologram.phase_pattern.shape)} (layer {args.layer_id}), distance {args.target_distance_mm} mm')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch, best_val, best_epoch = 1, float('inf'), 0
    best_val_psnr: float | None = None
    best_val_baseline_psnr: float | None = None
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if not args.reset_optimizer_on_resume:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resumed_epoch = int(checkpoint.get('epoch', 0))
        start_epoch = resumed_epoch + 1
        best_val = float(checkpoint.get('val_loss', best_val))
        best_epoch = int(checkpoint.get('best_epoch', 0))
        if checkpoint.get('best_val_psnr') is not None and not math.isnan(checkpoint['best_val_psnr']):
            best_val_psnr = float(checkpoint['best_val_psnr'])
        if checkpoint.get('best_val_dummy_psnr') is not None and not math.isnan(checkpoint['best_val_dummy_psnr']):
            best_val_baseline_psnr = float(checkpoint['best_val_dummy_psnr'])
        if args.extra_epochs > 0:
            args.epochs = resumed_epoch + args.extra_epochs
        elif args.epochs <= resumed_epoch:
            raise ValueError(f'--epochs ({args.epochs}) must be > the resumed epoch ({resumed_epoch}); or use --extra-epochs.')
        print(f'[Resume] {args.resume_checkpoint}: continuing at epoch {start_epoch} of {args.epochs}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = Path(args.viz_dir) if args.viz_dir else output_dir / 'viz_snapshots'
    run_config = {**vars(args), 'phase_file': str(args.phase_file), 'device': str(device)}
    (output_dir / 'config.json').write_text(json.dumps(run_config, indent=2))
    print(f'[Run] Output directory: {output_dir}')

    writer = None
    if args.tensorboard:
        tb_dir = Path(args.tb_dir) if args.tb_dir else output_dir / 'tensorboard'
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f'TensorBoard logging enabled at: {tb_dir}')

    global_step, snap_idx = 0, 0

    def snapshot(epoch: int) -> None:
        nonlocal snap_idx
        viz_dir.mkdir(parents=True, exist_ok=True)
        save_snapshot(model, hologram, val_loader, device, viz_dir, snap_idx, epoch, global_step, args.viz_max_images, viz_image_indices)
        snap_idx += 1

    if args.viz_every_steps > 0 and start_epoch == 1:
        print('[Viz] Saving pre-training snapshot (epoch 0, before any optimizer step)...')
        snapshot(0)

    for epoch in tqdm.trange(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running, samples = 0.0, 0
        stop = False
        for step, images in enumerate(train_loader, start=1):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.l1_loss(hologram(model, images), images)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.shape[0]
            samples += images.shape[0]
            global_step += 1

            if args.viz_every_steps > 0 and global_step % args.viz_every_steps == 0:
                snapshot(epoch)
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f'[Stop] Reached max optimizer steps ({args.max_steps}).')
                stop = True
                break
            if args.log_every > 0 and (step % args.log_every == 0 or step == len(train_loader)):
                print(f'[Train] Epoch {epoch:03d} Step {step:04d}/{len(train_loader):04d} loss={loss.item():.6f} avg_loss={running / samples:.6f}')
        train_loss = running / max(samples, 1)

        validate = epoch % args.val_every == 0
        if validate:
            val_loss, val_psnr, val_baseline_loss, val_baseline_psnr = evaluate(hologram, model, val_loader, device)
            val_log = (f'val_loss={val_loss:.6f} val_loss_dummy={val_baseline_loss:.6f} '
                       f'val_psnr={val_psnr:.2f}dB val_psnr_dummy={val_baseline_psnr:.2f}dB')
        else:
            val_loss = val_psnr = val_baseline_loss = val_baseline_psnr = float('nan')
            val_log = f'validation=skipped (runs every {args.val_every} epochs)'
        epoch_seconds = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        if writer is not None:
            writer.add_scalar('loss/train', train_loss, epoch)
            if validate:
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar('loss/val_dummy', val_baseline_loss, epoch)
                writer.add_scalar('psnr/val', val_psnr, epoch)
                writer.add_scalar('psnr/val_dummy', val_baseline_psnr, epoch)
            writer.add_scalar('optimizer/lr', lr, epoch)
            if epoch % args.tb_image_every == 0:
                reconstructions, lae_outputs = tensorboard_mosaics(hologram, model, val_loader, device, args)
                writer.add_image('visuals/input_dummy_ours_target', reconstructions, epoch)
                writer.add_image('visuals/raw_amp_and_model_output', lae_outputs, epoch)
            writer.flush()

        print(f'[Epoch] {epoch:03d}/{args.epochs:03d} train_loss={train_loss:.6f} {val_log} lr={lr:.2e} time={epoch_seconds:.1f}s')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_dummy_psnr': val_baseline_psnr,
            'best_val_psnr': best_val_psnr,
            'best_val_dummy_psnr': best_val_baseline_psnr,
            'best_epoch': best_epoch,
            'config': run_config,
        }
        torch.save(checkpoint, output_dir / 'last.pt')
        if validate and val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_val_psnr, best_val_baseline_psnr = val_psnr, val_baseline_psnr
            checkpoint.update(best_val_psnr=best_val_psnr, best_val_dummy_psnr=best_val_baseline_psnr, best_epoch=best_epoch)
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f'[Checkpoint] New best model saved at epoch {epoch:03d} (val_loss={val_loss:.6f}, '
                  f'val_psnr={val_psnr:.2f}dB, dummy_psnr={val_baseline_psnr:.2f}dB)')
        if stop:
            break

    if writer is not None:
        writer.close()
    gain = best_val_psnr - best_val_baseline_psnr if best_val_psnr is not None and best_val_baseline_psnr is not None else None
    (output_dir / 'sweep_metrics.json').write_text(json.dumps({
        'target_distance_mm': args.target_distance_mm,
        'model': 'lae',
        'tmx': args.tmx,
        'layer_id': args.layer_id,
        'total_epochs': args.epochs,
        'best_epoch': best_epoch,
        'best_val_loss': best_val if best_val != float('inf') else None,
        'best_val_psnr': best_val_psnr,
        'best_val_dummy_psnr': best_val_baseline_psnr,
        'psnr_gain': gain,
    }, indent=2))
    print(f'[Done] Training finished. sweep_metrics.json written to {output_dir}.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group('data')
    data.add_argument('--datasets-root', default=str(DATASET_2D_DIR), help='Folder holding the train/val image folders.')
    data.add_argument('--train-subdir', default='DIV2K_train_HR')
    data.add_argument('--val-subdir', default='DIV2K_valid_HR')
    data.add_argument('--image-rows', type=int, default=R, help='Images are resized to rows x cols (default: SLM size).')
    data.add_argument('--image-cols', type=int, default=C)
    data.add_argument('--num-workers', type=int, default=2)
    data.add_argument('--debug-single-sample', action='store_true', help='Train and validate on one training image.')
    data.add_argument('--debug-sample-index', type=int, default=0)

    optics = parser.add_argument_group('optics')
    optics.add_argument('--tmx', type=int, default=1, help='Number of time-multiplexed frames.')
    optics.add_argument('--target-distance-mm', type=float, default=100.0, help='Image plane to SLM distance.')
    optics.add_argument('--phase-file', type=Path, default=PHASE_FILE_2D, help='[LAYERS, TMX, RGB, H, W] random phase tensor.')
    optics.add_argument('--layer-id', type=int, default=0, help='Which layer of the phase tensor to use.')

    training = parser.add_argument_group('training')
    training.add_argument('--batch-size', type=int, default=8)
    training.add_argument('--epochs', type=int, default=30)
    training.add_argument('--lr', type=float, default=1e-4)
    training.add_argument('--weight-decay', type=float, default=1e-6)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Pick the GPU with CUDA_VISIBLE_DEVICES.')
    training.add_argument('--max-steps', type=int, default=0, help='Stop after this many optimizer steps (0: no limit).')
    training.add_argument('--resume-checkpoint', default='', help='Checkpoint (.pt) to resume from.')
    training.add_argument('--extra-epochs', type=int, default=0, help='With --resume-checkpoint: train this many more epochs.')
    training.add_argument('--reset-optimizer-on-resume', action='store_true')

    logging = parser.add_argument_group('outputs and logging')
    logging.add_argument('--output-dir', default=str(OUTPUT_DIR / '2d' / 'runs' / 'lae_2d'))
    logging.add_argument('--log-every', type=int, default=50, help='Print the training loss every N steps (0: off).')
    logging.add_argument('--val-every', type=int, default=1, help='Validate every N epochs.')
    logging.add_argument('--tensorboard', action='store_true')
    logging.add_argument('--tb-dir', default=None, help='Default: <output-dir>/tensorboard.')
    logging.add_argument('--tb-image-every', type=int, default=1, help='Log image mosaics every N epochs.')
    logging.add_argument('--tb-max-images', type=int, default=4)
    logging.add_argument('--tb-preview-rows', type=int, default=256)
    logging.add_argument('--tb-preview-cols', type=int, default=384)
    logging.add_argument('--viz-dir', default='', help='Default: <output-dir>/viz_snapshots.')
    logging.add_argument('--viz-every-steps', type=int, default=0,
                         help='Save an LAE/reconstruction/PSNR snapshot every N steps and before training (0: off).')
    logging.add_argument('--viz-max-images', type=int, default=4, help='Validation images saved per snapshot.')
    logging.add_argument('--viz-image-indices', default='', help='Comma-separated validation indices to save instead (e.g. "0,2").')
    args = parser.parse_args()
    if args.val_every < 1:
        parser.error('--val-every must be >= 1')
    return args


if __name__ == '__main__':
    train(parse_args())
