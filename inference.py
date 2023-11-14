import time
import os
import torch
import torch.nn as nn
from skimage.transform import resize
import numpy as np
import argparse
import cv2

from collections import OrderedDict
import utils.transforms as trans
from utils.configer import Configer
from utils.logger import Logger as Log

from models.attention_unet import AttU_Net
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas_Image


# activation = nn.LogSoftmax(dim=1)

def softmax(X, axis=0):
    max_prob = np.max(X, axis=axis, keepdims=True)
    X -= max_prob
    X = np.exp(X)
    sum_prob = np.sum(X, axis=axis, keepdims=True)
    X /= sum_prob
    return X

class ThermSeg():
    def __init__(self, configer):
        self.configer = configer
        self.seg_net = None
        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.NormalizeThermal(norm_mode=self.configer.get('normalize', 'norm_mode')), ])

        self.img_width, self.img_height = self.configer.get('data', 'input_size')

        if self.configer.get('gpu') is None:
            self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')


    def load_model(self, configer): 
        self.seg_net = AttU_Net(configer)
        self.seg_net = self.seg_net.to(self.device)
        self.seg_net.float()

        Log.info('Loading checkpoint from {}...'.format(self.configer.get('network', 'resume')))
        resume_dict = torch.load(self.configer.get('network', 'resume'), map_location=lambda storage, loc: storage)
        if 'state_dict' in resume_dict:
            checkpoint_dict = resume_dict['state_dict']

        elif 'model' in resume_dict:
            checkpoint_dict = resume_dict['model']

        elif isinstance(resume_dict, OrderedDict):
            checkpoint_dict = resume_dict

        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(self.configer.get('network', 'resume')))

        if list(checkpoint_dict.keys())[0].startswith('module.'):
            checkpoint_dict = {k[7:]: v for k, v in checkpoint_dict.items()}

        # load state_dict
        if hasattr(self.seg_net, 'module'):
            self.load_state_dict(self.seg_net.module, checkpoint_dict, self.configer.get('network', 'resume_strict'))
        else:
            self.load_state_dict(self.seg_net, checkpoint_dict, self.configer.get('network', 'resume_strict'))

        self.seg_net.eval()

        # load weights to device
        dummy_input_img = torch.ones((1, 1, self.img_width, self.img_height))
        dummy_input_img = dummy_input_img.to(self.device)
        logits = self.seg_net.forward(dummy_input_img)


    @staticmethod
    def load_state_dict(module, state_dict, strict=False):
        """Load state_dict to a module.
        This method is modified from :meth:`torch.nn.Module.load_state_dict`.
        Default value for ``strict`` is set to ``False`` and the message for
        param mismatch will be shown even if strict is False.
        Args:
            module (Module): Module that receives the state_dict.
            state_dict (OrderedDict): Weights.
            strict (bool): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        """
        unexpected_keys = []
        own_state = module.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception:
                Log.warn('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(),
                                           param.size()))
                
        missing_keys = set(own_state.keys()) - set(state_dict.keys())

        err_msg = []
        if unexpected_keys:
            err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
        if missing_keys:
            # we comment this to fine-tune the models with some missing keys.
            err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
        err_msg = '\n'.join(err_msg)
        if err_msg:
            if strict:
                raise RuntimeError(err_msg)
            else:
                Log.warn(err_msg)


    def delete_model(self):
        if self.seg_net != None:
            del self.seg_net
            self.seg_net = None

    def run_inference(self, input_img):

        t1 = time.time()
        with torch.no_grad():
            input_img = self.img_transform(input_img)
            input_img = input_img.unsqueeze(0)
            # input_img = self.module_runner.to_device(input_img)
            input_img = input_img.to(self.device)
            
            logits = self.seg_net.forward(input_img)
            if self.configer.get('gpu') is not None:
                torch.cuda.synchronize()
            # pred_mask = torch.argmax(activation(logits).exp(), dim=1).squeeze().cpu().numpy()
            
            logits = logits.permute(0, 2, 3, 1).cpu().numpy().squeeze()
            pred_mask = np.argmax(softmax(logits, axis=-1), axis=-1)
            time_taken = time.time() - t1

        return pred_mask, time_taken


def main(args_parser):
    configer = Configer(args_parser=args_parser)
    ckpt_root = configer.get('checkpoints', 'checkpoints_dir')
    ckpt_name = configer.get('checkpoints', 'checkpoints_name')
    configer.update(['network', 'resume'], os.path.join(ckpt_root, ckpt_name + '.pth'))
    thermal_file_ext = "bin"
    segObj = ThermSeg(configer)
    segObj.load_model(configer)

    datadir = args_parser.datadir
    fp_list = os.listdir(datadir)
    outdir = args_parser.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir_vis = os.path.join(args_parser.outdir, "vis")
    if not os.path.exists(outdir_vis):
        os.makedirs(outdir_vis)

    for fp in fp_list:
        fpath = os.path.join(datadir, fp)
        fp_ext = os.path.basename(fpath).split(".")[-1]
        if thermal_file_ext in fp_ext:
            Log.info('Processing: {}'.format(fp))
            try:
                thermal_matrix = np.fromfile(
                    fpath, dtype=np.uint16, count=segObj.img_width * segObj.img_height).reshape(segObj.img_height, segObj.img_width)
                thermal_matrix = (thermal_matrix  * 0.04) - 273.15
                pred_seg_mask, time_taken = segObj.run_inference(thermal_matrix)
            except Exception as e:
                Log.error('Error: {}'.format(e))

            fp_mask = os.path.join(outdir, os.path.basename(fpath).replace(".bin", ".png"))
            cv2.imwrite(fp_mask, pred_seg_mask)

            fp_mask_vis = os.path.join(outdir_vis, os.path.basename(fpath).replace(".bin", ".jpg"))
            fig = Figure(tight_layout=True)
            canvas = FigureCanvas_Image(fig)
            ax = fig.add_subplot(111)
            if np.all(pred_seg_mask) != None:
                ax.imshow(thermal_matrix, cmap='gray')
                ax.imshow(pred_seg_mask, cmap='seismic', alpha=0.35)
            else:
                ax.imshow(thermal_matrix, cmap='magma')
            ax.set_axis_off()
            canvas.draw()
            canvas.print_jpg(fp_mask_vis)

            Log.info('Done, inference time = {}.'.format(time_taken))


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=None,  type=str,
                        dest='datadir', help='The path containing raw thermal images.')
    parser.add_argument('--outdir', default=None,  type=str,
                        dest='outdir', help='The path to save the segmentation mask.')
    parser.add_argument('--configs', default=None,  type=str,
                        dest='configs', help='The path to congiguration file.')
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')
    parser.add_argument('--gathered', type=str2bool, nargs='?', default=True,
                        dest='network:gathered', help='Whether to gather the output of model.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network:resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network:resume_continue', help='Whether to continue training.')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()
    main(args_parser)