import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
from tqdm import tqdm
from monai.metrics import SSIMMetric


def relativistic_average_loss(real_logits, fake_logits):
    real_loss = torch.mean(F.softplus(-(real_logits - torch.mean(fake_logits))))
    fake_loss = torch.mean(F.softplus(-(fake_logits - torch.mean(real_logits))))
    return 0.5 * (real_loss + fake_loss)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        train_loader,
        val_loader,
        device,
        lr,
        beta1,
        beta2,
        checkpoint_dir,
    ):
        self.gen  = generator.to(device)
        self.disc = discriminator.to(device)
        self.tl   = train_loader
        self.vl   = val_loader
        self.dev  = device
        self.ckpt = checkpoint_dir

        self.opt_g = optim.Adam(self.gen.parameters(),  lr=lr, betas=(beta1, beta2))
        self.opt_d = optim.Adam(self.disc.parameters(), lr=lr, betas=(beta1, beta2))
        self.l1    = nn.L1Loss()
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_g, mode="min", factor=0.5, patience=200, verbose=True
        )

        self.ssim = SSIMMetric(data_range=1.0, spatial_dims=3, reduction="mean")

        os.makedirs(self.ckpt, exist_ok=True)
        wandb.log({
            "model/generator_params":     count_parameters(self.gen),
            "model/discriminator_params": count_parameters(self.disc),
        })

        self._train_losses = []
        self._val_losses   = []
        self._epochs       = []

    def train(self, epochs, return_metrics=False):
        best_val = float("inf")
        best_mae = best_psnr = best_ssim = None

        for ep in range(1, epochs + 1):
            self.gen.train()
            self.disc.train()
            g_losses = []
            d_losses = []

            for ct, mri, _ in tqdm(self.tl, desc=f"Epoch {ep} [Train]"):
                ct, mri = ct.to(self.dev), mri.to(self.dev)

                fake = self.gen(ct)
                d_loss = relativistic_average_loss(
                    self.disc(mri), self.disc(fake.detach())
                )
                d_losses.append(d_loss.item())
                self.opt_d.zero_grad()
                d_loss.backward()
                self.opt_d.step()

                fake   = self.gen(ct)
                gan    = relativistic_average_loss(self.disc(fake), self.disc(mri))
                l1     = self.l1(fake, mri)
                g_loss = gan + 100 * l1
                self.opt_g.zero_grad()
                g_loss.backward()
                self.opt_g.step()
                g_losses.append(g_loss.item())

            val_loss, mae, psnr, ssim = self._validate()
            self.sched.step(val_loss)

            if val_loss < best_val:
                best_val, best_mae, best_psnr, best_ssim = val_loss, mae, psnr, ssim
                torch.save(self.gen.state_dict(), os.path.join(self.ckpt, "best_generator.pth"))

            if ep % 500 == 0:
                torch.save(self.gen.state_dict(), os.path.join(self.ckpt, f"generator_epoch_{ep}.pth"))

            wandb.log({
                "loss/train_disc": np.mean(d_losses),
                "loss/train_gen":  np.mean(g_losses),
                "loss/val":        val_loss,
                "metrics/mae":     mae,
                "metrics/psnr":    psnr,
                "metrics/ssim":    ssim,
                "epoch":           ep,
                "lr":              self.opt_g.param_groups[0]["lr"],
            })

            self._train_losses.append(np.mean(g_losses))
            self._val_losses.append(val_loss)
            self._epochs.append(ep)

            if ep % 50 == 0:
                wandb.log({
                    "Overfitting Curve": wandb.plot.line_series(
                        xs=self._epochs,
                        ys=[self._train_losses, self._val_losses],
                        keys=["Generator Train Loss", "Validation Loss"],
                        title="Over-fitting Curve",
                        xname="Epoch",
                    )
                })

            print(
                f"Epoch {ep}: train_disc={np.mean(d_losses):.4f} | "
                f"train_gen={np.mean(g_losses):.4f} | val={val_loss:.4f} | "
                f"MAE={mae:.4f} | PSNR={psnr:.2f} | SSIM={ssim:.4f}"
            )

        if return_metrics:
            return best_mae, best_psnr, best_ssim

    def _validate(self):
        self.gen.eval()
        self.ssim.reset()
        losses, maes, psnrs, ssims = [], [], [], []

        with torch.no_grad():
            for ct, mri, mask in tqdm(self.vl, desc="[Validation]"):
                ct, mri = ct.to(self.dev), mri.to(self.dev)
                mask_b  = (mask > 0.5).to(self.dev)

                pred = self.gen(ct)
                gan  = relativistic_average_loss(self.disc(pred), self.disc(mri))
                l1   = self.l1(pred, mri)
                losses.append((gan + 100 * l1).item())

                p = pred.squeeze(1).cpu().numpy()
                g = mri.squeeze(1).cpu().numpy()
                m = mask_b.squeeze(1).cpu().numpy().astype(bool)

                if m.sum() == 0:
                    continue

                p[~m] = 0; g[~m] = 0
                flat_p, flat_g = p[m], g[m]
                maes.append(float(np.mean(np.abs(flat_p - flat_g))))

                mse = float(np.mean((flat_p - flat_g)**2))
                dr  = float(flat_g.max() - flat_g.min())
                if dr < 1e-8:
                    continue
                psnr = 10 * np.log10((dr**2) / (mse + 1e-12))
                psnrs.append(psnr)

                pred_masked = pred * mask_b
                gt_masked   = mri  * mask_b
                ssim_val    = self.ssim(pred_masked, gt_masked)
                if ssim_val.numel() > 1:
                    ssim_val = ssim_val.mean()
                ssims.append(ssim_val.item())

        return (
            np.mean(losses),
            np.nanmean(maes),
            np.nanmean(psnrs),
            np.nanmean(ssims),
        )