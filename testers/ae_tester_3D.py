import os
import tqdm
import importlib
from testers.base_tester import BaseTester
from testers.utils.utils import get_opt, get_prior, set_random_seed
from wt import *

try:
    from evaluation.evaluation_metrics import EMD_CD

    eval_reconstruciton = True
except:
    eval_reconstruciton = False
# from evaluation_metrics import EMD_CD

class Tester(BaseTester):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.tester, "seed", 666))

        # The networks
        sn_lib = importlib.import_module(cfg.models.scorenet.type)
        self.score_net = sn_lib.Decoder(cfg, cfg.models.scorenet)
        self.score_net.cuda()

        encoder_lib = importlib.import_module(cfg.models.encoder.type)
        self.encoder = encoder_lib.Encoder(cfg.models.encoder)
        self.encoder.cuda()

        # The optimizer
        if not (hasattr(self.cfg.tester, "opt_enc") and hasattr(self.cfg.tester, "opt_dec")):
            self.cfg.tester.opt_enc = self.cfg.tester.opt
            self.cfg.tester.opt_dec = self.cfg.tester.opt

        self.opt_enc, self.scheduler_enc = get_opt(
            self.encoder.parameters(), self.cfg.tester.opt_enc)
        self.opt_dec, self.scheduler_dec = get_opt(
            self.score_net.parameters(), self.cfg.tester.opt_dec)

        # Sigmas
        if hasattr(cfg.tester, "sigmas"):
            self.sigmas = cfg.tester.sigmas
        else:
            self.sigma_begin = float(cfg.tester.sigma_begin)
            self.sigma_end = float(cfg.tester.sigma_end)
            self.num_classes = int(cfg.tester.sigma_num)
            self.sigmas = np.exp(
                np.linspace(np.log(self.sigma_begin),
                            np.log(self.sigma_end),
                            self.num_classes))

        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)

        # Prepare variable for summy
        self.oracle_res = None


    def validate(self, test_loader,  epoch, *args, **kwargs):
        if not eval_reconstruciton:
            return {}

        print("Validation reconstruction:")
        all_ref, all_rec, all_smp, all_ref_denorm = [], [], [], []
        all_rec_gt, all_inp_denorm, all_inp = [], [], []

        for data in tqdm.tqdm(test_loader):
            ref_pts = data['te_points'].cuda()
            inp_pts = data['tr_points'].cuda()
            m = data['mean'].cuda()
            std = data['std'].cuda()

            rec_pts,_ = self.reconstruct(inp_pts.cuda(), num_points=inp_pts.size(1))

            inp_pts = idwt_3D(inp_pts)
            ref_pts_denorm = ref_pts.clone() * std + m
            inp_pts_denorm = inp_pts.clone() * std + m
            rec_pts = rec_pts * std + m

            all_inp.append(inp_pts)
            all_inp_denorm.append(inp_pts_denorm.view(*inp_pts.size()))
            all_ref_denorm.append(ref_pts_denorm.view(*ref_pts.size()))
            all_rec.append(rec_pts.view(*ref_pts.size()))
            all_ref.append(ref_pts)

        inp = torch.cat(all_inp, dim=0)
        rec = torch.cat(all_rec, dim=0)
        ref = torch.cat(all_ref, dim=0)
        ref_denorm = torch.cat(all_ref_denorm, dim=0)
        inp_denorm = torch.cat(all_inp_denorm, dim=0)
        for name, arr in [
            ('inp', inp), ('rec', rec), ('ref', ref),
            ('ref_denorm', ref_denorm), ('inp_denorm', inp_denorm)]:
            np.save(
                os.path.join(
                    self.cfg.save_dir, 'val', '%s_ep%d.npy' % (name, epoch)),
                arr.detach().cpu().numpy()
            )
        all_res = {}


        # Oracle CD/EMD, will compute only once
        if self.oracle_res is None:
            rec_res = EMD_CD(inp_denorm, ref_denorm, 1)
            rec_res = {
                ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
                for k, v in rec_res.items()}
            all_res.update(rec_res)
            print("Validation oracle (denormalize) Epoch:%d " % epoch, rec_res)
            self.oracle_res = rec_res
        else:
            all_res.update(self.oracle_res)

        all_res = {}
        rec_res = EMD_CD(rec, ref_denorm, 1)
        rec_res = {
            ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
            for k, v in rec_res.items()}
        all_res.update(rec_res)
        print("Validation Recon (denormalize) Epoch:%d " % epoch, rec_res)

        return all_res

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_enc': self.opt_enc.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            'sn': self.score_net.state_dict(),
            'enc': self.encoder.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.score_net.load_state_dict(ckpt['sn'], strict=strict)
        self.opt_enc.load_state_dict(ckpt['opt_enc'])
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def langevin_dynamics(self, z, num_points=1024):
        with torch.no_grad():
            assert hasattr(self.cfg, "inference")
            step_size_ratio = float(getattr(self.cfg.inference, "step_size_ratio", 1))
            num_steps = int(getattr(self.cfg.inference, "num_steps", 5))
            num_points = int(getattr(self.cfg.inference, "num_points", num_points))

            weight = float(getattr(self.cfg.inference, "weight", 1))
            sigmas = self.sigmas

            x_list = []
            self.score_net.eval()

            x = get_prior(z.size(0), num_points, self.cfg.models.scorenet.dim)
            x = x.to(z)
            x0 = x.clone()
            x_list.append(x0)

            for sigma in sigmas:
                sigma = torch.ones((1,)) * sigma
                sigma = sigma.cuda()
                z_sigma = torch.cat((z, sigma.expand(z.size(0), 1)), dim=1)

                step_size = 2 * sigma ** 2 * step_size_ratio
                for t in range(num_steps):
                    noise = torch.sqrt(step_size) * torch.randn_like(x0) * weight
                    x0 += noise
                    grad = self.score_net(x0, z_sigma)
                    grad = grad / sigma ** 2
                    x0 += 0.5 * grad * step_size
                x = idwt_3D(x0)
                x_list.append(x0.clone())
        return x, x_list

    def sample(self, num_shapes=1, num_points=1024):
        with torch.no_grad():
            z = torch.randn(num_shapes, self.cfg.models.encoder.zdim).cuda()
            return self.langevin_dynamics(z, num_points=num_points)

    def reconstruct(self, inp_pts, num_points=1024):
        with torch.no_grad():
            self.encoder.eval()
            z, _ = self.encoder(inp_pts)
            return self.langevin_dynamics(z, num_points=num_points)