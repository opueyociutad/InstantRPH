#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

End-to-end training of Instant RPH (the 3D LAE) against target focal stacks.
Multi-GPU through accelerate (--use-accelerate, launched with `accelerate launch`).
Train one wavelength per run (--train-wavelength r/g/b) to save memory,
then merge the three runs with merge_rgb_3d.py.
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from instantrph.common import (
    DATASET_3D_DIR, OUTPUT_DIR, PHASE_FILE_3D, WAVELENGTH_CHANNELS, focus_distances_m, load_phase, pick_device, prepare_volume,
    seed_all, select_stack_channels, to_rgb_stack,
)
from instantrph.data_3d import MPIFocalStackDataset, collate_scene_batch, split_dataset_indices
from instantrph.metrics import colorbalance
from instantrph.models import LinearAmplitudeEncoder3D
from instantrph.renderer_3d import LayeredFocalStackRenderer, evaluate
from instantrph.viz import focal_stack_mosaic, mpi_data_preview


def make_dataloaders(args: argparse.Namespace, num_workers: int) -> tuple[DataLoader, DataLoader | None, int]:
    """Seeded train/val split (only the training scenes are augmented). Returns (train, val, number of scenes)."""
    def dataset(augment: bool) -> MPIFocalStackDataset:
        return MPIFocalStackDataset(
            args.data_root, input_is_intensity=not args.input_is_amplitude, enable_augmentation=augment,
            flip_h_prob=args.flip_h_prob, flip_v_prob=args.flip_v_prob,
        )

    base = dataset(augment=False)
    if len(base) == 0:
        raise ValueError(f'No scene pairs found under {args.data_root}')
    if args.debug_single_sample:
        train_set = val_set = Subset(base, [args.debug_sample_index % len(base)])
    else:
        train_indices, val_indices = split_dataset_indices(len(base), val_fraction=args.val_fraction, seed=args.seed)
        train_set = Subset(dataset(augment=args.enable_augmentation), train_indices)
        val_set = Subset(base, val_indices) if val_indices else None

    def loader(subset: Subset, shuffle: bool) -> DataLoader:
        return DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(), collate_fn=collate_scene_batch)

    val_loader = loader(val_set, False) if val_set is not None else None
    return loader(train_set, not args.debug_single_sample), val_loader, len(base)


def first_batches(loader: DataLoader, max_images: int) -> dict[str, torch.Tensor] | None:
    """Up to max_images samples (across batches) for the TensorBoard previews."""
    chunks: dict[str, list[torch.Tensor]] = {'rgb_vol': [], 'alpha_vol': [], 'focal_stack': []}
    remaining = max_images
    for batch in loader:
        if remaining <= 0:
            break
        take = min(remaining, batch['rgb_vol'].shape[0])
        for key in chunks:
            chunks[key].append(batch[key][:take])
        remaining -= take
    return {key: torch.cat(value) for key, value in chunks.items()} if chunks['rgb_vol'] else None


def train(args: argparse.Namespace) -> None:
    accelerator = None
    if args.use_accelerate:
        from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
        from accelerate.utils import set_seed

        # Cluster startup and the first collectives can be slow; avoid watchdog timeouts.
        accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(), InitProcessGroupKwargs(timeout=timedelta(minutes=30))])
        device = accelerator.device
        set_seed(args.seed)
    else:
        seed_all(args.seed)
        device = pick_device(args.device)
    is_main = accelerator is None or accelerator.is_main_process

    channels = WAVELENGTH_CHANNELS[args.train_wavelength]
    train_loader, val_loader, num_scenes = make_dataloaders(args, num_workers=args.num_workers)
    first = train_loader.dataset[0]
    num_layers, image_rows, image_cols = first['rgb_vol'].shape[1:]
    focus_count = first['focal_stack'].shape[0]
    focus_m = focus_distances_m(focus_count, start_mm=args.focus_start_mm, end_mm=args.focus_end_mm,
                                center_mm=args.focus_center_mm, distances_mm=args.focus_distances_mm)

    model = LinearAmplitudeEncoder3D(tmx=args.tmx, depth=num_layers, height=image_rows, width=image_cols, num_rgb_channels=len(channels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_val = 1, float('inf')
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if not args.reset_optimizer_on_resume:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        best_val = float(checkpoint.get('best_val_loss', checkpoint.get('val_loss', best_val)))
        print(f'[Resume] {args.resume_checkpoint}: continuing at epoch {start_epoch}')
    if accelerator is None:
        model = model.to(device)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    unwrapped = accelerator.unwrap_model(model) if accelerator else model

    phase = load_phase(args.phase_file, tmx=args.tmx, channels=channels, image_hw=(image_rows, image_cols))
    renderer = LayeredFocalStackRenderer(
        image_rows=image_rows, image_cols=image_cols, layer_spacing_m=args.layer_spacing_mm / 1000.0,
        focus_distances_m=focus_m, phase_tensor=phase.to(device), tmx=args.tmx, layer_id=args.phase_layer_id,
        wavelength_indices=channels,
    )

    output_dir = Path(args.output_dir)
    config = {
        **vars(args),
        'phase_file': str(args.phase_file),
        'device': str(device),
        'dataset_size': num_scenes,
        'focus_distances_m': focus_m,
        'train_wavelength_channels': ['RGB'[c] for c in channels],
    }
    writer = None
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'config.json').write_text(json.dumps(config, indent=2))
        print(f'[Init] Device: {device}' + (f' ({accelerator.num_processes} processes)' if accelerator else ''))
        print(f'[Init] Scenes: {config["dataset_size"]} | validation: {val_loader is not None}')
        print(f'[Init] MPI (D, H, W): ({num_layers}, {image_rows}, {image_cols}) | focus planes (mm): {[round(d * 1000, 4) for d in focus_m]}')
        print(f'[Init] Wavelengths: {config["train_wavelength_channels"]} | tmx: {args.tmx} | phase: {tuple(phase.shape)}')
        print(f'[Init] Trainable parameters: {sum(p.numel() for p in model.parameters()):,}')
        if args.tensorboard:
            tb_dir = Path(args.tb_dir) if args.tb_dir else output_dir / 'tensorboard'
            writer = SummaryWriter(log_dir=str(tb_dir))
            print(f'TensorBoard logging enabled at: {tb_dir}')
            if args.visualize_training_data:
                for b, batch in zip(range(args.viz_data_samples), train_loader):
                    for i, name in enumerate(batch['scene_name'][:4]):
                        writer.add_image(f'data/batch{b:02d}/{name}', mpi_data_preview(batch['rgb_vol'][i], batch['focal_stack'][i]), 0)
    preview_batch = first_batches(val_loader or train_loader, args.tb_max_images) if writer is not None else None

    def validate() -> tuple[float, float]:
        loss, psnr_sum, count = evaluate(model, renderer, val_loader, channels, device)
        if accelerator is not None:  # sum over processes
            loss, psnr_sum, count = (accelerator.gather(torch.tensor(v, device=device)).sum().item() for v in (loss, psnr_sum, count))
        return loss / max(count, 1), psnr_sum / max(count, 1)

    def log_preview(step: int) -> None:
        if preview_batch is None or step % args.tb_image_every != 0:
            return
        rgb_vol, alpha_vol = prepare_volume(preview_batch['rgb_vol'], preview_batch['alpha_vol'], device, channels)
        target_stack = select_stack_channels(preview_batch['focal_stack'].to(device=device, dtype=torch.float32), channels)
        with torch.no_grad():
            pred_stack = colorbalance(renderer.render_stack(model(rgb_vol), alpha_vol), target_stack).clamp(0.0, 1.0)
        mosaic = focal_stack_mosaic(to_rgb_stack(pred_stack), to_rgb_stack(target_stack), renderer.focus_distances_m,
                                    max_images=args.tb_max_images, size=(args.tb_preview_rows, args.tb_preview_cols))
        writer.add_image('visuals/target_then_prediction', mosaic, step)

    if val_loader is not None:
        val_loss, val_psnr = validate()
        if is_main:
            print(f'[Epoch] {start_epoch - 1:03d}/{args.epochs:03d} train_loss=nan train_psnr=nan val_loss={val_loss:.6f} val_psnr={val_psnr:.2f}dB (pre-train)')
        if writer is not None:
            writer.add_scalar('loss/val', val_loss, start_epoch - 1)
            writer.add_scalar('psnr/val', val_psnr, start_epoch - 1)
            writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], start_epoch - 1)
            log_preview(start_epoch - 1)
            writer.flush()

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        running_loss = running_psnr = 0.0
        seen = 0
        for batch in tqdm(train_loader, disable=not is_main):
            rgb_vol, alpha_vol = prepare_volume(batch['rgb_vol'], batch['alpha_vol'], device, channels)
            target_stack = select_stack_channels(batch['focal_stack'].to(device=device, dtype=torch.float32), channels)
            optimizer.zero_grad(set_to_none=True)
            loss, psnr_value = renderer.render_stack_loss(model(rgb_vol), alpha_vol, target_stack)
            accelerator.backward(loss) if accelerator else loss.backward()
            if args.grad_clip > 0:
                params = [p for group in optimizer.param_groups for p in group['params']]
                (accelerator.clip_grad_norm_ if accelerator else torch.nn.utils.clip_grad_norm_)(params, args.grad_clip)
            optimizer.step()
            running_loss += loss.item() * rgb_vol.shape[0]
            running_psnr += psnr_value.item() * rgb_vol.shape[0]
            seen += rgb_vol.shape[0]
        train_loss, train_psnr = running_loss / max(seen, 1), running_psnr / max(seen, 1)
        val_loss, val_psnr = validate() if val_loader is not None else (float('nan'), float('nan'))
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else None

        if writer is not None:
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('psnr/train', train_psnr, epoch)
            if val_loader is not None:
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar('psnr/val', val_psnr, epoch)
            writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)
            if peak_mb is not None:
                writer.add_scalar('memory/max_allocated_mb', peak_mb, epoch)
            log_preview(epoch)
            writer.flush()

        if is_main:
            improved = val_loader is not None and val_loss < best_val
            best_val = val_loss if improved else best_val
            checkpoint = {
                'epoch': epoch, 'model_state_dict': unwrapped.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss, 'val_loss': val_loss, 'best_val_loss': best_val, 'config': config,
            }
            torch.save(checkpoint, output_dir / 'last.pt')
            if improved:
                torch.save(checkpoint, output_dir / 'best.pt')
                print(f'[Checkpoint] Saved best.pt at epoch {epoch:03d} (val_loss={val_loss:.6f}, val_psnr={val_psnr:.2f}dB)')
            peak_text = f' peak_mem={peak_mb:.2f}MB' if peak_mb is not None else ''
            print(f'[Epoch] {epoch:03d}/{args.epochs:03d} train_loss={train_loss:.6f} train_psnr={train_psnr:.2f}dB '
                  f'val_loss={val_loss:.6f} val_psnr={val_psnr:.2f}dB{peak_text}')

    if writer is not None:
        writer.close()
    if is_main:
        print('[Done] 3D training finished.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group('data')
    data.add_argument('--data-root', default=str(DATASET_3D_DIR), help='Folder with MPI / focal-stack scene pairs (see instantrph/data_3d.py).')
    data.add_argument('--val-fraction', type=float, default=0.1)
    data.add_argument('--input-is-amplitude', action='store_true', help='MPI colors are amplitudes (default: intensities, square-rooted).')
    data.add_argument('--enable-augmentation', action='store_true', help='Random flips and RGB permutations of training scenes.')
    data.add_argument('--flip-h-prob', type=float, default=0.5)
    data.add_argument('--flip-v-prob', type=float, default=0.5)
    data.add_argument('--num-workers', type=int, default=2, help='Data-loading workers per process.')
    data.add_argument('--debug-single-sample', action='store_true', help='Train and validate on one scene.')
    data.add_argument('--debug-sample-index', type=int, default=0)

    optics = parser.add_argument_group('optics')
    optics.add_argument('--tmx', type=int, default=8, help='Number of time-multiplexed frames.')
    optics.add_argument('--train-wavelength', type=str.lower, default='rgb', choices=list(WAVELENGTH_CHANNELS))
    optics.add_argument('--phase-file', type=Path, default=PHASE_FILE_3D, help='[LAYERS, TMX, RGB, H, W] random phase tensor.')
    optics.add_argument('--phase-layer-id', type=int, default=0, help='Phase layer of the nearest MPI layer.')
    optics.add_argument('--layer-spacing-mm', type=float, default=0.25, help='MPI layer spacing (selects the data/models_3d/kernels/ RSD kernel).')
    optics.add_argument('--focus-start-mm', type=float, default=95.0, help='Focal stack: linear sweep start...')
    optics.add_argument('--focus-end-mm', type=float, default=101.0, help='...and end.')
    optics.add_argument('--focus-center-mm', type=float, default=100.0, help='Focus distance of single-plane stacks.')
    optics.add_argument('--focus-distances-mm', default='', help='Comma-separated focus distances (overrides the sweep).')

    training = parser.add_argument_group('training')
    training.add_argument('--epochs', type=int, default=40)
    training.add_argument('--batch-size', type=int, default=1)
    training.add_argument('--lr', type=float, default=1e-4)
    training.add_argument('--weight-decay', type=float, default=1e-6)
    training.add_argument('--grad-clip', type=float, default=1.0)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Ignored with --use-accelerate.')
    training.add_argument('--use-accelerate', action='store_true', help='Distributed training (launch with `accelerate launch`).')
    training.add_argument('--resume-checkpoint', default='', help='A last.pt / best.pt to resume from.')
    training.add_argument('--reset-optimizer-on-resume', action='store_true')

    logging = parser.add_argument_group('outputs and logging')
    logging.add_argument('--output-dir', default=str(OUTPUT_DIR / '3d' / 'runs' / 'lae_3d'))
    logging.add_argument('--tensorboard', action='store_true')
    logging.add_argument('--tb-dir', default=None, help='Default: <output-dir>/tensorboard.')
    logging.add_argument('--tb-image-every', type=int, default=1, help='Log focal-stack previews every N epochs.')
    logging.add_argument('--tb-max-images', type=int, default=2)
    logging.add_argument('--tb-preview-rows', type=int, default=800)
    logging.add_argument('--tb-preview-cols', type=int, default=1280)
    logging.add_argument('--visualize-training-data', action='store_true', help='Log a few training MPIs and focal stacks.')
    logging.add_argument('--viz-data-samples', type=int, default=2, help='Training batches to log with --visualize-training-data.')
    args = parser.parse_args()
    if args.tb_image_every < 1:
        parser.error('--tb-image-every must be >= 1')
    return args


if __name__ == '__main__':
    train(parse_args())
