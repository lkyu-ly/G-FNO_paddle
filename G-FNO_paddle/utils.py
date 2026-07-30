import paddle


class pde_data(paddle.io.Dataset):
    def __init__(self, data, T_in, T_out=None, train=True, strategy="markov", std=0.0):
        self.markov = strategy == "markov"
        self.teacher_forcing = strategy == "teacher_forcing"
        self.one_shot = strategy == "oneshot"
        self.data = (
            data[..., : T_in + T_out] if self.one_shot else data[..., : T_in + T_out, :]
        )
        self.nt = T_in + T_out
        self.T_in = T_in
        self.T_out = T_out
        self.num_hist = 1 if self.markov else self.T_in
        self.train = train
        self.noise_std = std

    def __len__(self):
        if self.train:
            if self.markov:
                return len(self.data) * (self.nt - 1)
            if self.teacher_forcing:
                return len(self.data) * (self.nt - self.T_in)
        return len(self.data)

    def __getitem__(self, idx):
        if not self.train or not (self.markov or self.teacher_forcing):
            pde = self.data[idx]
            if self.one_shot:
                x = pde[..., : self.T_in, :]
                x = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                y = pde[..., self.T_in : self.T_in + self.T_out, :]
            else:
                x = pde[..., self.T_in - self.num_hist : self.T_in, :]
                y = pde[..., self.T_in : self.T_in + self.T_out, :]
            return x, y
        pde_idx = idx // (self.nt - self.num_hist)
        t_idx = idx % (self.nt - self.num_hist) + self.num_hist
        pde = self.data[pde_idx]
        x = pde[..., t_idx - self.num_hist : t_idx, :]
        y = pde[..., t_idx, :]
        if self.noise_std > 0:
            x += paddle.randn(*x.shape, device=x.device) * self.noise_std
        return x, y


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = h ** (self.d / self.p) * paddle.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        if self.reduction:
            if self.size_average:
                return paddle.mean(all_norms)
            else:
                return paddle.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        assert x.shape == y.shape and len(x.shape) == 3, "wrong shape"
        diff_norms = paddle.norm(x - y, self.p, 1)
        y_norms = paddle.norm(y, self.p, 1)
        if self.reduction:
            loss = (diff_norms / y_norms).mean(-1)
            if self.size_average:
                return paddle.mean(loss)
            else:
                return paddle.sum(loss)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def eq_check_rt(model, x, spatial_dims):
    model.eval()
    diffs = []
    with paddle.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in range(len(spatial_dims)):
            for l in range(j + 1, len(spatial_dims)):
                dims = [spatial_dims[j], spatial_dims[l]]
                diffs.append(
                    [
                        (
                            (
                                (
                                    out.rot90(k=k, axes=dims)
                                    - model(x.rot90(k=k, axes=dims))
                                )
                                / out.rot90(k=k, axes=dims)
                            )
                            .abs()
                            .nanmean()
                            .item()
                            * 100
                        )
                        for k in range(1, 4)
                    ]
                )
    return paddle.tensor(diffs).mean().item()


def eq_check_rf(model, x, spatial_dims):
    model.eval()
    diffs = []
    with paddle.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in spatial_dims:
            diffs.append(
                ((out.flip(axis=(j,)) - model(x.flip(axis=(j,)))) / out.flip(axis=(j,)))
                .abs()
                .nanmean()
                .item()
                * 100
            )
    return paddle.tensor(diffs).mean().item()


class grid(paddle.nn.Module):
    def __init__(self, twoD, grid_type):
        super(grid, self).__init__()
        assert grid_type in ["cartesian", "symmetric", "None"], "Invalid grid type"
        self.symmetric = grid_type == "symmetric"
        self.include_grid = grid_type != "None"
        self.grid_dim = (1 + (not self.symmetric) + (not twoD)) * self.include_grid
        if self.include_grid:
            if twoD:
                self.get_grid = self.twoD_grid
            else:
                self.get_grid = self.threeD_grid
        else:
            self.get_grid = paddle.nn.Identity()

    def forward(self, x):
        return self.get_grid(x)

    def twoD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = paddle.tile(
            paddle.linspace(0, 1, size_x).reshape([1, size_x, 1, 1]),
            [batchsize, 1, size_y, 1],
        )
        gridy = paddle.tile(
            paddle.linspace(0, 1, size_y).reshape([1, 1, size_y, 1]),
            [batchsize, size_x, 1, 1],
        )
        if not self.symmetric:
            grid = paddle.cat((gridx, gridy), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = gridx + gridy
        grid = grid.to(x.device)
        return paddle.cat((x, grid), dim=-1)

    def threeD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = (
            paddle.linspace(0, 1, size_x)
            .reshape(1, size_x, 1, 1, 1)
            .repeat([batchsize, 1, size_y, size_z, 1])
        )
        gridy = (
            paddle.linspace(0, 1, size_y)
            .reshape(1, 1, size_y, 1, 1)
            .repeat([batchsize, size_x, 1, size_z, 1])
        )
        gridz = (
            paddle.linspace(0, 1, size_z)
            .reshape(1, 1, 1, size_z, 1)
            .repeat([batchsize, size_x, size_y, 1, 1])
        )
        if not self.symmetric:
            grid = paddle.cat((gridx, gridy, gridz), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = paddle.cat((gridx + gridy, gridz), dim=-1)
        grid = grid.to(x.device)
        return paddle.cat((x, grid), dim=-1)
