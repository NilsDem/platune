import gin
import torch
import torch.nn as nn
import torch.distributions as D
import pytorch_lightning as pl

from typing import Callable, List, Tuple, Dict

from platune.helpers.data_visualization import plot_features_extraction


@gin.configurable
class PLaTune(pl.LightningModule):

    def __init__(
        self,
        flow: Callable[[], nn.Module],
        latent_dim: int = 64,
        discrete_keys: List[str] = None,
        continuous_keys: List[str] = None,
        classes_attr_discrete: List[List[int]] = [],
        min_max_attr_continuous: List[Tuple[int]] = [],
        label_dict: Dict[str, int] = {},
        bins_values: List[List[int]] = [],
        sigma_init: float = 0.4,
        r: float = 0.25,
        sigma_decay: float = 0.995,
        sigma_target_continuous: float = 0.005,
        lr: int = 1e-4,
        use_grad_clip: bool = False,
        n_ex_val: int = 0,
        nb_steps: int = 20,
        scale_controls: float = 1.,
        emb_model=None,
    ):

        super().__init__()

        # model
        self.flow = flow()
        self.emb_model = [emb_model]
        self.scale_controls = scale_controls

        # discrete controls
        self.discrete_keys = discrete_keys
        self.n_attr_discrete = len(self.discrete_keys)
        self.classes_attr_discrete = classes_attr_discrete
        self.n_classes = [len(a) for a in self.classes_attr_discrete
                          ] if self.n_attr_discrete > 0 else []
        self.min_max_attr_discrete = [(0, v - 1) for v in self.n_classes
                                      ] if self.n_attr_discrete > 0 else []

        # label controls
        self.label_dict = label_dict
        self.n_labels = len(
            self.label_dict) if self.label_dict is not None else 0

        # continuous controls
        self.continuous_keys = continuous_keys
        self.n_attr_continuous = len(self.continuous_keys)
        self.min_max_attr_continuous = min_max_attr_continuous

        self.bins_values = torch.tensor(bins_values)
        self.n_quantized_classes = None
        if len(bins_values) > 0:
            print(
                'using quantized continuous attributes (e.g. processed by the model as discrete attributes)'
            )
            self.n_quantized_classes = len(bins_values[0])
            self.min_max_attr = self.min_max_attr_discrete + [
                (0, self.n_quantized_classes - 1)
                for _ in range(self.n_attr_continuous)
            ]
        else:
            self.min_max_attr = self.min_max_attr_discrete + self.min_max_attr_continuous

        if self.n_labels:
            self.min_max_attr.append((0, self.n_labels - 1))

        self.all_keys = self.discrete_keys + self.continuous_keys
        self.all_keys += ["instrument"] if self.n_labels > 0 else []

        # dims
        self.latent_dim = latent_dim
        self.control_dim = self.n_attr_discrete + self.n_attr_continuous
        self.control_dim += (1 if self.n_labels > 0 else 0)
        self.style_dim = latent_dim - self.control_dim

        # hparams
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.r = r

        sigma_target = []
        if self.n_attr_discrete > 0:
            self.sigma_target_discrete = torch.tensor([
                (2 / self.n_classes[i]) * r
                for i in range(self.n_attr_discrete)
            ])
            sigma_target.append(self.sigma_target_discrete)

        if self.n_attr_continuous > 0:
            if len(bins_values) > 0:
                self.sigma_target_continuous = torch.tensor([
                    (2 / self.n_quantized_classes) * r
                    for i in range(self.n_attr_continuous)
                ])
            else:
                self.sigma_target_continuous = torch.full(
                    (self.n_attr_continuous, ), sigma_target_continuous)

            sigma_target.append(self.sigma_target_continuous)

        if self.n_labels > 0:
            self.sigma_target_label = torch.tensor([(2 / self.n_labels) * r
                                                    for i in range(1)])

            sigma_target.append(self.sigma_target_label)

        self.sigma_target = torch.cat(sigma_target)

        self.lr = lr
        self.use_grad_clip = use_grad_clip

        self.automatic_optimization = False

        # for validation step
        self.n_examples = n_ex_val
        self.nb_steps = nb_steps
        self.validation_step_outputs = {}

    def configure_optimizers(self):
        params = list(self.flow.parameters())
        opt = torch.optim.AdamW(params, self.lr, (.5, .999))
        return opt

    def normalize_attr(self, x, invert=False):
        min_v = [t[0] for t in self.min_max_attr]
        min_values = torch.tensor(min_v).unsqueeze(0).unsqueeze(2).to(x.device)

        max_v = [t[1] for t in self.min_max_attr]
        max_values = torch.tensor(max_v).unsqueeze(0).unsqueeze(2).to(x.device)

        if not invert:
            # normalize between -1 and 1
            x = (x - min_values) / (max_values - min_values)
            x = 2 * x - 1
            x = self.scale_controls * x
        else:
            x = x / self.scale_controls
            x = (0.5 * (x + 1))
            x = (max_values - min_values) * x + min_values

        return x

    def convert_class_values_to_indices(self, a):
        new_a = []
        for i in range(self.n_attr_discrete):
            classes_i = torch.tensor(self.classes_attr_discrete[i]).to(
                a.device)
            a_ids = torch.searchsorted(classes_i.contiguous(),
                                       a[:, i, :].contiguous())
            new_a.append(a_ids.unsqueeze(1))
        new_a = torch.cat(new_a, dim=1)
        return new_a

    def process_attributes(self, ad, ac, label):
        attr = []
        if ad.shape[-1] > 0:
            ad = ad.to(self.device)
            ad_ids = self.convert_class_values_to_indices(ad)
            attr.append(ad_ids)

        if ac.shape[-1] > 0:
            ac = ac.to(self.device)
            if len(self.bins_values) > 0:
                ac_quantized = torch.zeros_like(ac).to(self.device)
                self.bins_values = self.bins_values.to(ac.device)
                for i in range(self.n_attr_continuous):
                    data = ac[:, i, :].flatten()
                    classes = torch.bucketize(data, self.bins_values[i])
                    ac_quantized[:, i, :] = classes.reshape(-1, ac.shape[-1])
                attr.append(ac_quantized)
            else:
                attr.append(ac)

        if len(self.label_dict) > 0:
            values = [self.label_dict[l] for l in label]
            values = torch.tensor(values).unsqueeze(1).to(self.device)
            values = values.reshape(-1, 1, 1).repeat(1, 1, attr[-1].shape[-1])
            attr.append(values)

        attr = torch.cat(attr, dim=1)

        return attr

    @torch.no_grad()
    def cs_to_z(self, cs, c=None, nb_steps=10):
        dt = 1 / nb_steps
        t_values = torch.linspace(0, 1, nb_steps + 1).to(self.device)[:-1]
        x = cs.to(self.device)

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x + self.flow(x, time=t) * dt
        return x

    @torch.no_grad()
    def z_to_cs(self, z, c=None, nb_steps=10):
        dt = 1 / nb_steps
        t_values = torch.linspace(1, 0, nb_steps + 1).to(self.device)[:-1]
        x = z.to(self.device)

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x - self.flow(x, time=t) * dt
        return x

    def edit(self, z, c, c_orig=None, nb_steps=10, warmup=True):
        cs_rec = self.z_to_cs(z, nb_steps=nb_steps)
        s = cs_rec[:, self.control_dim:, :]

        c_dist, _ = self.get_cs_distributions(c, warmup=warmup)

        c_samples = c_dist.sample()
        cs = torch.cat([c_samples, s], dim=1)

        z_edit = self.cs_to_z(cs, nb_steps=nb_steps)

        return z_edit

    def get_sigma(self, a, warmup=False):
        if warmup:
            # apply warmup on sigma_target
            progress = self.global_step / 50
            current_sigma_values = self.sigma_target + (
                self.sigma_init - self.sigma_target) * (self.sigma_decay**
                                                        progress)
        else:
            current_sigma_values = self.sigma_target

        current_sigma = torch.cat([
            torch.full((a.shape[0], 1, a.shape[-1]), current_sigma_values[i])
            for i in range(self.control_dim)
        ],
                                  dim=1).to(a.device)
        return current_sigma

    def get_cs_distributions(self, a, warmup=False, zero_var=False):
        # define control distribution
        current_sigma = self.get_sigma(a, warmup)

        if zero_var:
            c_dist = D.Normal(a, 0.001 * torch.ones_like(current_sigma))
        else:
            c_dist = D.Normal(a, current_sigma)

        # define style distribution
        s_dist = D.Normal(
            torch.zeros(
                (a.shape[0], self.style_dim, a.shape[-1])).to(a.device),
            torch.ones((a.shape[0], self.style_dim, a.shape[-1])).to(a.device))

        return c_dist, s_dist

    def get_cs_samples(self, c_dist, s_dist):
        c_samples = c_dist.sample()
        s_samples = s_dist.sample()
        cs = torch.cat([c_samples, s_samples], dim=1)
        return cs

    def compute_nll(self, cs, c_dist, s_dist):
        c = cs[:, :self.control_dim]
        s = cs[:, self.control_dim:]

        c_logp = c_dist.log_prob(c)
        s_logp = s_dist.log_prob(s)

        log_likelihood = ((c_logp.sum(1, keepdim=True) +
                           s_logp.sum(1, keepdim=True))).mean()

        return -log_likelihood

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()

        z, ad, ac, label = batch
        z = z.to(self.device)
        a = self.process_attributes(ad, ac, label)
        c = self.normalize_attr(a)

        # Get the distribution samples
        c_dist, s_dist = self.get_cs_distributions(c, warmup=True)
        cs = self.get_cs_samples(c_dist, s_dist)

        # diffusion loss
        target = z - cs
        t = torch.rand(z.size(0), 1, 1).to(self.device)
        interpolant = (1 - t) * cs + t * z
        model_output = self.flow(interpolant, time=t)
        loss = ((model_output - target)**2).mean()

        # optimization
        opt.zero_grad()
        loss.backward()
        # gradient clipping:
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(list(self.flow.parameters()), 1.)
        opt.step()

        # tensorboard
        self.log("diffusion_loss", loss)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.trainer is not None:

            z, ad, ac, label = batch
            z = z.to(self.device)
            a = self.process_attributes(ad, ac, label)
            c = self.normalize_attr(a)

            c_dist, s_dist = self.get_cs_distributions(c, warmup=False)
            cs = self.get_cs_samples(c_dist, s_dist)

            cs_rec = self.z_to_cs(z, c, nb_steps=self.nb_steps)
            nll_loss = self.compute_nll(cs_rec, c_dist, s_dist)

            z_rec = self.cs_to_z(cs, c=c, nb_steps=self.nb_steps)
            z_loss = torch.nn.functional.mse_loss(z, z_rec)

            # control distributions with zero sigma (almost equivalent to taking attributes values directly)
            c_dist0, s_dist0 = self.get_cs_distributions(c, zero_var=True)
            cs0 = self.get_cs_samples(c_dist0, s_dist0)
            z_rec0 = self.cs_to_z(cs0, c=c, nb_steps=self.nb_steps)
            z_loss_zerosigma = torch.nn.functional.mse_loss(z, z_rec0)

            self.validation_step_outputs[
                "feat_extract_loss"] = self.validation_step_outputs.get(
                    "feat_extract_loss", []) + [nll_loss.item()]
            self.validation_step_outputs[
                "rec_loss"] = self.validation_step_outputs.get(
                    "rec_loss", []) + [z_loss.item()]
            self.validation_step_outputs[
                "rec_loss_zerosigma"] = self.validation_step_outputs.get(
                    "rec_loss_zerosigma", []) + [z_loss_zerosigma.item()]

            if self.n_examples > 0:
                ex_c_gt = self.validation_step_outputs.get("c_gt", [])
                if len(ex_c_gt) == 0 or len(ex_c_gt) < self.n_examples:
                    n_ex = self.n_examples - len(ex_c_gt)
                    curr_c_gt = [c[i] for i in range(n_ex)]
                    self.validation_step_outputs["c_gt"] = ex_c_gt + curr_c_gt
                    curr_c_rec = [
                        cs_rec[:, :self.control_dim, :][i] for i in range(n_ex)
                    ]
                    self.validation_step_outputs[
                        "c_rec"] = self.validation_step_outputs.get(
                            "c_rec", []) + curr_c_rec

            self.log('validation', nll_loss.item())

            if batch_idx == 0:  # log only once per validation
                print("Loading transfers")
                ## TRANSFER ##
                # Swapping control across batch
                idx = torch.randperm(z.size(0))
                c_swapped = c[idx]
                z_transfer = self.edit(z[:4],
                                       c=c_swapped[:4],
                                       c_orig=c[:4],
                                       nb_steps=self.nb_steps)
                z_target = z[idx]

                with torch.no_grad():

                    audio_original = self.emb_model[0].decode(
                        z[:4].cpu()).squeeze()
                    audio_rec = self.emb_model[0].decode(
                        z_rec[:4].cpu()).squeeze()
                    audio_rec0 = self.emb_model[0].decode(
                        z_rec0[:4].cpu()).squeeze()
                    audio_transfer = self.emb_model[0].decode(
                        z_transfer[:4].cpu()).squeeze()
                    audio_targets = self.emb_model[0].decode(
                        z_target[:4].cpu()).squeeze()

                for i in range(len(audio_original)):
                    self.logger.experiment.add_audio(
                        tag=f"reconstruction/original_{i}",
                        snd_tensor=audio_original[i],
                        sample_rate=44100,
                        global_step=self.global_step,
                    )
                    self.logger.experiment.add_audio(
                        tag=f"reconstruction/rec_{i}",
                        snd_tensor=audio_rec[i],
                        sample_rate=44100,
                        global_step=self.global_step,
                    )
                    self.logger.experiment.add_audio(
                        tag=f"reconstruction/rec_zero_sigma_{i}",
                        snd_tensor=audio_rec0[i],
                        sample_rate=44100,
                        global_step=self.global_step,
                    )

                    self.logger.experiment.add_audio(
                        tag=f"transfer/transfer_{i}",
                        snd_tensor=audio_transfer[i],
                        sample_rate=44100,
                        global_step=self.global_step,
                    )
                    self.logger.experiment.add_audio(
                        tag=f"transfer/transfer_targets_{i}",
                        snd_tensor=audio_targets[i],
                        sample_rate=44100,
                        global_step=self.global_step,
                    )
                print("Finished loading transfers")

            return nll_loss.item()

    def on_validation_epoch_end(self):

        for k, v in self.validation_step_outputs.items():
            if k not in ["c_gt", "c_rec"]:
                self.log(k, torch.mean(torch.tensor(v)))

        n_ex = len(self.validation_step_outputs["c_gt"])
        if n_ex > 0:
            for i in range(n_ex):
                for k in range(self.control_dim):
                    f_ik = plot_features_extraction(
                        c_gt=self.validation_step_outputs["c_gt"][i][k, :],
                        c_rec=self.validation_step_outputs["c_rec"][i][k, :],
                        descriptor_name=self.all_keys[k],
                        figsize=(10, 5),
                    )
                    self.logger.experiment.add_figure(
                        f"ex {i} - features extraction c_{k}", f_ik,
                        self.global_step)

        self.validation_step_outputs = {}


@gin.configurable
class SDEdit(PLaTune):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @torch.no_grad()
    def cs_to_z(self, cs, c, nb_steps=10, x0=None):
        dt = 1 / nb_steps
        t_values = torch.linspace(0, 1, nb_steps + 1).to(self.device)[:-1]

        if x0 is None:
            x = torch.randn_like(cs).to(self.device)
        else:
            x = x0

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x + self.flow(x, time=t, time_cond=c) * dt
        return x

    @torch.no_grad()
    def z_to_cs(self, z, c, nb_steps=10):
        dt = 1 / nb_steps
        t_values = torch.linspace(1, 0, nb_steps + 1).to(self.device)[:-1]
        x = z.to(self.device)

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x - self.flow(x, time=t, time_cond=c) * dt
        return x

    def edit(self, z, c, c_orig, nb_steps=10):
        x0 = self.z_to_cs(z, c=c_orig, nb_steps=nb_steps)
        z_edit = self.cs_to_z(None, c=c, x0=x0, nb_steps=nb_steps)
        return z_edit

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()

        z, ad, ac, label = batch
        z = z.to(self.device)
        a = self.process_attributes(ad, ac, label)
        c = self.normalize_attr(a)

        # Get the distribution samples
        # c_dist, s_dist = self.get_cs_distributions(c, warmup=True)
        # cs = self.get_cs_samples(c_dist, s_dist)
        cs = torch.randn_like(z)
        # diffusion loss
        target = z - cs
        t = torch.rand(z.size(0), 1, 1).to(self.device)
        interpolant = (1 - t) * cs + t * z
        model_output = self.flow(interpolant, time=t, time_cond=c)
        loss = ((model_output - target)**2).mean()

        # optimization
        opt.zero_grad()
        loss.backward()
        # gradient clipping:
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(list(self.flow.parameters()), 1.)
        opt.step()

        # tensorboard
        self.log("diffusion_loss", loss)
