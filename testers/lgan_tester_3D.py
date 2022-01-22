import os
import tqdm
import torch
import random
import importlib
import numpy as np
from testers.utils.utils import get_opt
from testers.ae_tester_3D import Tester as BaseTester

try:
    from evaluation.evaluation_metrics import compute_all_metrics
    eval_generation = True
except:
    eval_generation = False

class Tester(BaseTester):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        # Now initialize the GAN part
        gen_lib = importlib.import_module(cfg.models.gen.type)
        self.gen = gen_lib.Generator(cfg, cfg.models.gen)
        self.gen.cuda()
        dis_lib = importlib.import_module(cfg.models.dis.type)
        self.dis = dis_lib.Discriminator(cfg, cfg.models.dis)
        self.dis.cuda()

        # Optimizers
        if not (hasattr(self.cfg.tester, "opt_gen") and
                hasattr(self.cfg.tester, "opt_dis")):
            self.cfg.tester.opt_gen = self.cfg.tester.opt
            self.cfg.tester.opt_dis = self.cfg.tester.opt
        self.opt_gen, self.scheduler_gen = get_opt(
            self.gen.parameters(), self.cfg.tester.opt_gen)
        self.opt_dis, self.scheduler_dis = get_opt(
            self.dis.parameters(), self.cfg.tester.opt_dis)

        # book keeping
        self.total_iters = 0
        self.total_gan_iters = 0
        self.n_critics = getattr(self.cfg.tester, "n_critics", 1)
        self.gan_only = getattr(self.cfg.tester, "gan_only", True)
        # If pretrained AE, then load it up
        if hasattr(self.cfg.tester, "ae_pretrained"):
            ckpt = torch.load(self.cfg.tester.ae_pretrained)
            strict = getattr(self.cfg.tester, "resume_strict", True)
            self.encoder.load_state_dict(ckpt['enc'], strict=strict)
            self.score_net.load_state_dict(ckpt['sn'], strict=strict)
            if getattr(self.cfg.tester, "resume_opt", False):
                self.opt_enc.load_state_dict(ckpt['opt_enc'])
                self.opt_dec.load_state_dict(ckpt['opt_dec'])
        self.gan_pass_update_enc = getattr(
            self.cfg.tester, "gan_pass_update_enc", False)

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_enc': self.opt_enc.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            'opt_dis': self.opt_dis.state_dict(),
            'opt_gen': self.opt_gen.state_dict(),
            'sn': self.score_net.state_dict(),
            'enc': self.encoder.state_dict(),
            'dis': self.dis.state_dict(),
            'gen': self.gen.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, **args):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.score_net.load_state_dict(ckpt['sn'], strict=strict)
        self.opt_enc.load_state_dict(ckpt['opt_enc'])
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']

        if 'gen' in ckpt:
            self.gen.load_state_dict(ckpt['gen'], strict=strict)
        if 'dis' in ckpt:
            self.dis.load_state_dict(ckpt['dis'], strict=strict)
        if 'opt_gen' in ckpt:
            self.opt_gen.load_state_dict(ckpt['opt_gen'])
        if 'opt_dis' in ckpt:
            self.opt_dis.load_state_dict(ckpt['opt_dis'])
        return start_epoch


    def sample(self, num_shapes = 1, num_points=2048):
        with torch.no_grad():
            self.gen.eval()
            z = self.gen(bs=num_shapes)
            return self.langevin_dynamics(z, num_points=num_points)

    def validate(self, test_loader, epoch, *args, **kwargs):
        all_res = {}
        if eval_generation:
            with torch.no_grad():
                print("l-GAN validation:")
                all_ref, all_smp = [], []
                for data in tqdm.tqdm(test_loader):
                    ref_pts = data['te_points'].cuda()
                    inp_pts = data['tr_points'].cuda()
                    smp_pts, _ = self.sample(
                        num_shapes=inp_pts.size(0),
                        num_points=inp_pts.size(1),
                    )

                    all_smp.append(smp_pts.view(
                        ref_pts.size(0), ref_pts.size(1), ref_pts.size(2)))
                    all_ref.append(
                        ref_pts.view(ref_pts.size(0), ref_pts.size(1),
                                     ref_pts.size(2)))

                smp = torch.cat(all_smp, dim=0)
                np.save(
                    os.path.join(self.cfg.save_dir, 'val',
                                 'smp_ep%d.npy' % epoch),
                    smp.detach().cpu().numpy()
                )
                ref = torch.cat(all_ref, dim=0)

                # Sample CD/EMD
                # step 1: subsample shapes
                max_gen_vali_shape = int(getattr(
                    self.cfg.tester, "max_gen_validate_shapes",
                    int(smp.size(0))))
                sub_sampled = random.sample(
                    range(smp.size(0)), min(smp.size(0), max_gen_vali_shape))

                smp_sub = smp[sub_sampled, ...].contiguous()
                ref_sub = ref[sub_sampled, ...].contiguous()
                gen_res = compute_all_metrics(
                    smp_sub, ref_sub,
                    batch_size=int(getattr(
                        self.cfg.tester, "val_metrics_batch_size", 100)),
                    accelerated_cd=True
                )
                all_res = {
                    ("val/gen/%s" % k):
                        (v if isinstance(v, float) else v.item())
                    for k, v in gen_res.items()}
                print("Validation Sample (unit) Epoch:%d " % epoch, gen_res)



        # Call super class validation
        if getattr(self.cfg.tester, "validate_recon", False):
            all_res.update(super().validate(
                test_loader, epoch, *args, **kwargs))

        return all_res