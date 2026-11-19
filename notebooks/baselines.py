import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def softmax_np(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def entropy_np(logits):
    p = softmax_np(logits)
    return -(p * np.log(p + 1e-12)).sum(1)


def coverage_and_size(sets, labels):
    return float(sets[np.arange(len(labels)), labels].mean()), float(sets.sum(1).mean())


def ecp_sets(cal_logits, cal_labels, test_logits, alpha, power=2):
    p_cal = softmax_np(cal_logits)
    n = len(cal_labels)
    tau = np.quantile(p_cal[np.arange(n), cal_labels], (1 + 1 / n) * alpha, method="midpoint")
    u = np.quantile(entropy_np(test_logits), 1 - alpha)
    return softmax_np(test_logits) * max(1.0, u) ** power >= tau


def weighted_quantile(values, weights, level):
    order = np.argsort(values)
    cum = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    i = int(np.searchsorted(cum, level * cum[-1], side="left"))
    return float(np.asarray(values)[order[min(i, len(values) - 1)]])


def wqlcp_sets(cal_logits, cal_labels, cal_loss, test_logits, test_loss, alpha):
    p_cal = softmax_np(cal_logits)
    r = 1 - p_cal[np.arange(len(cal_labels)), cal_labels]
    tau = 1 - weighted_quantile(r, cal_loss, 1 - alpha)
    scale = max(1.0, np.quantile(test_loss, 1 - alpha))
    return softmax_np(test_logits) * scale >= tau


class SmallVAE(nn.Module):
    def __init__(self, channels, size, latent=32):
        super().__init__()
        self.side = size // 4
        flat = 64 * self.side * self.side
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent)
        self.logvar = nn.Linear(256, latent)
        self.decoder_fc = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, flat), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x, sample=True):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return self.decoder(self.decoder_fc(z).view(-1, 64, self.side, self.side)), mu, logvar


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_vae(dataset, device, seed, epochs=100, batch_size=256, lr=1e-4, weight_decay=0.01, beta=1.2):
    _seed_all(seed)
    x0 = dataset[0][0]
    vae = SmallVAE(x0.shape[0], x0.shape[-1]).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        _seed_all(seed + epoch)
        vae.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                            generator=torch.Generator().manual_seed(seed + epoch))
        for x, _ in loader:
            x = x.to(device)
            rec, mu, logvar = vae(x)
            sse = (rec - x).square().flatten(1).sum(1).mean()
            kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)).mean()
            opt.zero_grad(set_to_none=True)
            (sse + beta * kl).backward()
            opt.step()
    return vae.eval()


@torch.no_grad()
def collect_logits_and_vae_loss(model, loader, sigma, shift, vae, mean, std, device):
    model.eval()
    m = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    s = torch.tensor(std, device=device).view(1, -1, 1, 1)
    logits, labels, losses = [], [], []
    for x, y in loader:
        x = shift(x.to(device), sigma)
        raw = x * s + m
        rec, _, _ = vae(raw, sample=False)
        logits.append(model.logits_from_x_nograd(x).double().cpu().numpy())
        labels.append(y.numpy())
        losses.append((rec - raw).square().flatten(1).sum(1).double().cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels), np.concatenate(losses)
