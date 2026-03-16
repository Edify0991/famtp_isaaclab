"""Train latent bridge generator from pseudo bridge tasks (no transition labels)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from famtp_lab.priors.bridge_generator import BridgeGeneratorCfg, LatentBridgeGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FaMTP bridge generator.")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--bridge-horizon", type=int, default=12)
    parser.add_argument("--latent-dim-total", type=int, default=20)
    parser.add_argument("--num-skills", type=int, default=3)
    parser.add_argument("--use-target-anchor", type=int, choices=[0, 1], default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _pseudo_bridge_batch(
    batch_size: int,
    history_steps: int,
    latent_dim_total: int,
    num_skills: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct pseudo-bridge tasks from end-of-A and start-of-B anchors.

    Returns:
        z_history: (B, H, Z)
        current_skill: (B,)
        target_skill: (B,)
        target_anchor: (B, Z)
    """
    current_skill = torch.randint(0, num_skills, (batch_size,), device=device)
    target_skill = torch.randint(0, num_skills, (batch_size,), device=device)
    z_end_a = torch.randn(batch_size, latent_dim_total, device=device)
    z_start_b = 0.5 * z_end_a + 0.5 * torch.randn(batch_size, latent_dim_total, device=device)
    z_history = z_end_a[:, None, :].expand(batch_size, history_steps, latent_dim_total) + 0.03 * torch.randn(
        batch_size, history_steps, latent_dim_total, device=device
    )
    return z_history, current_skill, target_skill, z_start_b


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = BridgeGeneratorCfg(
        latent_dim_total=args.latent_dim_total,
        history_steps=args.history_steps,
        horizon_steps=args.bridge_horizon,
        hidden_dim=128,
        num_skills=args.num_skills,
        use_target_anchor=bool(args.use_target_anchor),
    )
    model = LatentBridgeGenerator(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    curves = {
        "terminal_anchor_consistency": [],
        "latent_smoothness": [],
        "global_coupling_consistency": [],
        "latent_prior_consistency": [],
        "contact_consistency": [],
        "total": [],
    }

    for step in range(args.steps):
        z_history, current_skill, target_skill, target_anchor = _pseudo_bridge_batch(
            args.batch_size, args.history_steps, args.latent_dim_total, args.num_skills, device
        )
        z_future = model(z_history, current_skill, target_skill, target_anchor=target_anchor)

        # 1) terminal anchor consistency
        loss_terminal = torch.mean((z_future[:, -1, :] - target_anchor) ** 2)
        # 2) latent smoothness
        loss_smooth = torch.mean((z_future[:, 1:, :] - z_future[:, :-1, :]) ** 2)
        # 3) global coupling consistency (proxy: small acceleration in global latent norm)
        norm = torch.norm(z_future, dim=-1)
        loss_global = torch.mean((norm[:, 2:] - 2 * norm[:, 1:-1] + norm[:, :-2]) ** 2) if z_future.shape[1] > 2 else 0.0
        # 4) latent prior consistency (proxy: keep magnitudes bounded)
        loss_prior = torch.mean(torch.relu(torch.abs(z_future) - 3.0) ** 2)
        # 5) optional contact consistency regularization (proxy binary-sparsity on first 4 dims)
        contact_logits = z_future[:, :, :4]
        loss_contact = torch.mean(torch.sigmoid(contact_logits) * (1.0 - torch.sigmoid(contact_logits)))

        total = (
            1.0 * loss_terminal
            + 0.3 * loss_smooth
            + 0.5 * torch.as_tensor(loss_global, device=device)
            + 0.5 * loss_prior
            + 0.1 * loss_contact
        )

        opt.zero_grad()
        total.backward()
        opt.step()

        curves["terminal_anchor_consistency"].append(float(loss_terminal.item()))
        curves["latent_smoothness"].append(float(loss_smooth.item()))
        curves["global_coupling_consistency"].append(float(torch.as_tensor(loss_global).item()))
        curves["latent_prior_consistency"].append(float(loss_prior.item()))
        curves["contact_consistency"].append(float(loss_contact.item()))
        curves["total"].append(float(total.item()))

        if step % 100 == 0:
            print(f"step={step} total={float(total.item()):.5f}")

    ckpt_dir = Path("outputs/checkpoints/bridge")
    log_dir = Path("outputs/logs/bridge")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "bridge_generator_latest.pt"
    torch.save({"model": model.state_dict(), "cfg": vars(args)}, ckpt_path)

    summary = {
        "checkpoint": str(ckpt_path),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "loss_curves": curves,
        "final_losses": {k: v[-1] for k, v in curves.items()},
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved bridge checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
