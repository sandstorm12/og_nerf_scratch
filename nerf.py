
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._linear_0 = nn.Linear(60, 256)
        self._linear_1 = nn.Linear(256, 256)
        self._linear_2 = nn.Linear(256, 256)
        self._linear_3 = nn.Linear(256, 256)
        self._linear_4 = nn.Linear(256 + 60, 256)
        self._linear_5 = nn.Linear(256, 256)
        self._linear_6 = nn.Linear(256, 256)
        self._linear_7 = nn.Linear(256, 256)

        self._output = nn.Linear(256, 4)

    def _positional_encoding(self, x, num_freqs):
        encoding = []
        for i in range(num_freqs):
            freq = 2.**i
            encoding.append(torch.sin(freq * x))
            encoding.append(torch.cos(freq * x))

        encoding = torch.stack(encoding, -1)

        return encoding

    def forward(self, x, y, z, phi, theta):
        x = self._positional_encoding(x, 10)
        y = self._positional_encoding(y, 10)
        z = self._positional_encoding(z, 10)
        # phi = self._positional_encoding(phi, 6)
        # theta = self._positional_encoding(theta, 6)

        location = torch.cat([x, y, z], -1)
        # direction = torch.cat([phi, theta], -1)

        t = F.relu_(self._linear_0(location))
        t = F.relu_(self._linear_1(t))
        t = F.relu_(self._linear_2(t))
        t = F.relu_(self._linear_3(t))
        t = torch.cat([t, location], -1)
        t = F.relu_(self._linear_4(t))
        t = F.relu_(self._linear_5(t))
        t = F.relu_(self._linear_6(t))
        t = F.relu_(self._linear_7(t))

        t = self._output(t)

        rgb = F.sigmoid(t[..., :3])
        a = F.softplus(t[..., 3])

        rgba = torch.cat([rgb, a[..., None]], -1)

        return rgba


if __name__ == "__main__":
    nerf = NeRF()
    
    x = torch.rand((4, 100), dtype=torch.float32) * 8 - 4
    y = torch.rand((4, 100), dtype=torch.float32) * 8 - 4
    z = torch.rand((4, 100), dtype=torch.float32) * 8 - 4
    phi = torch.rand((4, 100), dtype=torch.float32)
    theta = torch.rand((4, 100), dtype=torch.float32)
    rgba = nerf(x, y, z, phi, theta)

    torch.save(nerf.state_dict(), "./artifacts/nerf_raw.pt")
