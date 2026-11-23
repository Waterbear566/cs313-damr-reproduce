"""Mạng và loss của DeepSVDD / ALOCC chép từ repo MTGFlow, dùng chung cho baselines/DeepSVDD và baselines/ALOCC.

Nguồn: https://github.com/zqhang/MTGFLOW/blob/f8cb9ea225fd7e64cfb4556dae3525accac285f5/models/GAN.py
(sha256 2ba491d5e94e880660ae584d6cac045d6f85dea9a864f4f2fab8a71bc04b5e5c), dòng 9-159.

Chép nguyên văn, chỉ khác:
- Bỏ import không dùng (`asyncio.log.logger`, `utils.log`, `timeit`) và các hàm Wasserstein không dùng.
- D_Net._compute_out_dim: bản gốc đưa `torch.Tensor(1, C, L)` (bộ nhớ CHƯA khởi tạo, có thể chứa NaN) qua
  BatchNorm ở train mode, làm bẩn running_mean/var trước khi train. Ở đây dùng tensor 0 ở eval mode rồi trả lại
  mode cũ. Kích thước đầu ra không đổi; không động tới tham số.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvEncoder(activation=nn.LeakyReLU, in_channels: int = 3, n_c: int = 64, k_size: int = 5):
    enc = nn.Sequential(*(nn.Conv1d(in_channels, n_c, k_size, stride=2, padding=2),
                          nn.BatchNorm1d(n_c),
                          activation(),
                          nn.Conv1d(n_c, n_c * 2, k_size, stride=2, padding=2),
                          nn.BatchNorm1d(n_c * 2),
                          activation(),
                          nn.Conv1d(n_c * 2, n_c * 4, k_size, stride=2, padding=2),
                          nn.BatchNorm1d(n_c * 4),
                          activation()))
    return enc


def ConvDecoder(activation=nn.LeakyReLU, in_channels: int = 3, n_c: int = 64, k_size: int = 5):
    decoder = nn.Sequential(*(nn.ConvTranspose1d(n_c * 4, n_c * 2, k_size, stride=2, padding=2, output_padding=0),
                              torch.nn.BatchNorm1d(n_c * 2),
                              activation(),
                              torch.nn.ConvTranspose1d(n_c * 2, n_c, k_size, stride=2, padding=2, output_padding=1),
                              torch.nn.BatchNorm1d(n_c),
                              activation(),
                              torch.nn.ConvTranspose1d(n_c, in_channels, k_size, stride=2, padding=2, output_padding=1)))
    return decoder


class CNNAE(torch.nn.Module):
    """Recurrent autoencoder"""

    def __init__(self, in_channels: int = 3, n_channels: int = 16, kernel_size: int = 5):
        super(CNNAE, self).__init__()
        activation = torch.nn.LeakyReLU
        self.in_channels = in_channels
        self.n_c = n_channels
        self.k_size = kernel_size
        self.encoder = ConvEncoder(activation, in_channels, n_channels, kernel_size)
        self.decoder = ConvDecoder(activation, in_channels, n_channels, kernel_size)

    def forward(self, x: torch.Tensor):
        z = self.encoder.forward(x)
        x_out = self.decoder.forward(z)
        return x_out


class R_Net(torch.nn.Module):

    def __init__(self, activation=torch.nn.LeakyReLU, in_channels: int = 3, n_channels: int = 16,
                 kernel_size: int = 5, std: float = 0.2):
        super(R_Net, self).__init__()
        self.activation = activation
        self.in_channels = in_channels
        self.n_c = n_channels
        self.k_size = kernel_size
        self.std = std
        self.Encoder = ConvEncoder(activation, in_channels, n_channels, kernel_size)
        self.Decoder = ConvDecoder(activation, in_channels, n_channels, kernel_size)

    def forward(self, x: torch.Tensor, noise: bool = True):
        x_hat = self.add_noise(x) if noise else x
        z = self.Encoder.forward(x_hat)
        x_out = self.Decoder.forward(z)
        return x_out

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.std
        x_hat = x + noise
        return x_hat


class D_Net(torch.nn.Module):

    def __init__(self, in_resolution: int, activation=torch.nn.LeakyReLU, in_channels: int = 3,
                 n_channels: int = 16, kernel_size: int = 5):
        super(D_Net, self).__init__()
        self.activation = activation
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.n_c = n_channels
        self.k_size = kernel_size
        self.cnn = ConvEncoder(activation, in_channels, n_channels, kernel_size)
        self.out_dim = self._compute_out_dim()
        self.fc = torch.nn.Linear(self.out_dim, 1)

    def _compute_out_dim(self):
        # KHÁC BẢN GỐC (xem docstring đầu file): tensor 0, eval mode, không cập nhật running stats.
        was_training = self.cnn.training
        self.cnn.eval()
        try:
            with torch.no_grad():
                test_x = self.cnn(torch.zeros(1, self.in_channels, self.in_resolution))
        finally:
            self.cnn.train(was_training)
        return torch.prod(torch.tensor(test_x.shape[1:])).item()

    def forward(self, x: torch.Tensor):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        return out


def R_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:
    pred = d_net(x_fake)
    y = torch.ones_like(pred)
    rec_loss = F.mse_loss(x_fake, x_real)
    gen_loss = F.binary_cross_entropy_with_logits(pred, y)  # generator loss
    L_r = gen_loss + lambd * rec_loss
    return {'rec_loss': rec_loss, 'gen_loss': gen_loss, 'L_r': L_r}


def D_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
    pred_real = d_net(x_real)
    pred_fake = d_net(x_fake.detach())
    y_real = torch.ones_like(pred_real)
    y_fake = torch.zeros_like(pred_fake)
    real_loss = F.binary_cross_entropy_with_logits(pred_real, y_real)
    fake_loss = F.binary_cross_entropy_with_logits(pred_fake, y_fake)
    return real_loss + fake_loss
