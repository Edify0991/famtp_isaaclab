"""Train part-wise manifold encoders for FaMTP Stage 1."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch
from torch import nn

from famtp_lab.priors.manifold_encoders import ManifoldEncoderCfg, MultiPartManifoldEncoder
from famtp_lab.priors.part_discriminators import PART_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train part manifold encoders.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--residual-dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sample_batch(batch_size: int, history_steps: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate synthetic proprio-history data for each part.

    Shape notes:
        out[name]: (B, H, 3)
    """
    out: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(PART_NAMES):
        t = torch.linspace(0.0, 1.0, history_steps, device=device)[None, :, None]
        phase = 0.6 * idx + torch.rand(batch_size, 1, 1, device=device)
        sig = torch.cat(
            [
                torch.sin(2.0 * torch.pi * (t + phase)),
                torch.cos(2.0 * torch.pi * (t + phase)),
                0.1 * torch.randn(batch_size, history_steps, 1, device=device),
            ],
            dim=-1,
        )
        out[name] = sig
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ManifoldEncoderCfg(history_steps=args.history_steps, part_input_dim=3, residual_dim=args.residual_dim, hidden_dim=64)
    model = MultiPartManifoldEncoder(cfg).to(device)

    pred_head = nn.Linear(model.latent_dim_per_part, 3).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(pred_head.parameters()), lr=args.lr)

    losses = []
    prev_theta: dict[str, torch.Tensor] | None = None
    for step in range(args.steps):
        hist = _sample_batch(args.batch_size, args.history_steps, device)
        out = model(hist)

        recon_loss = torch.tensor(0.0, device=device)
        phase_smooth_loss = torch.tensor(0.0, device=device)
        temporal_consistency_loss = torch.tensor(0.0, device=device)
        propagation_loss = torch.tensor(0.0, device=device)

        for part in PART_NAMES:
            # short-horizon prediction: predict last frame feature from latent z.
            z = out["z_by_part"][part]  # (B, 2+R)
            pred = pred_head(z)  # (B, 3)
            target = hist[part][:, -1, :]  # (B, 3)
            recon_loss = recon_loss + torch.mean((pred - target) ** 2)

            # phase smoothness: penalize d(theta)^2 between consecutive mini-batch calls.
            theta = out["theta_by_part"][part]  # (B, 1)
            if prev_theta is not None:
                phase_smooth_loss = phase_smooth_loss + torch.mean((theta - prev_theta[part]) ** 2)

            # latent temporal consistency in history: residual norms should stay bounded.
            residual = z[:, 2:]
            temporal_consistency_loss = temporal_consistency_loss + torch.mean(residual**2)

            # optional latent propagation consistency: z_t approximately predicts z_t+1 trend.
            z_next_est = torch.roll(z, shifts=-1, dims=0)
            propagation_loss = propagation_loss + torch.mean((z - z_next_est) ** 2)

        total = recon_loss + 0.1 * phase_smooth_loss + 0.05 * temporal_consistency_loss + 0.02 * propagation_loss

        opt.zero_grad()
        total.backward()
        opt.step()

        prev_theta = {part: out["theta_by_part"][part].detach() for part in PART_NAMES}
        losses.append(float(total.item()))

        if step % 100 == 0:
            print(f"step={step} loss={float(total.item()):.6f}")

    ckpt_dir = Path("outputs/checkpoints/part_encoders")
    log_dir = Path("outputs/logs/part_encoders")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "part_encoder_latest.pt"
    torch.save({"model": model.state_dict(), "pred_head": pred_head.state_dict(), "cfg": vars(args)}, ckpt_path)

    summary = {
        "checkpoint": str(ckpt_path),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "history_steps": args.history_steps,
        "residual_dim": args.residual_dim,
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
    }
    (log_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved encoder checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
