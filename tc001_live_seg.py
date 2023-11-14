import cv2
import numpy as np
import argparse
from inference import ThermSeg
from utils.configer import Configer
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



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
    parser.add_argument('--device', type=str, dest='device', default=2, help='camera device number')
    parser.add_argument('--configs', default='configs/AU_SAMCL.json',  type=str,
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


    cap = cv2.VideoCapture(
        '/dev/video'+str(args_parser.device), cv2.CAP_V4L)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

    configer = Configer(args_parser=args_parser)
    ckpt_root = configer.get('checkpoints', 'checkpoints_dir')
    ckpt_name = configer.get('checkpoints', 'checkpoints_name')
    configer.update(['network', 'resume'], os.path.join(ckpt_root, ckpt_name + '.pth'))
    segObj = ThermSeg(configer)
    segObj.load_model(configer)


    width = 256  # Sensor width
    height = 192  # sensor height

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(figsize=(8,8))
    init_image1 = (np.random.rand(height, width) + 1) * 20
    init_image2 = (np.random.rand(height, width) + 1) * 3
    im1 = ax.imshow(init_image1, cmap='gray', vmin=20, vmax=40)
    im2 = ax.imshow(init_image2, cmap='seismic', alpha=0.5, vmin=0, vmax=5)


    # fps = 25
    # nSeconds = 10

    def init():
        ax.set_axis_off()
        return im1, im2


    def live_capture(i):

        if (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                _, thdata = np.array_split(frame, 2)
                hi = thdata[..., 0]
                lo = thdata[..., 1]
                lo = lo * 256.0
                raw_temp = hi + lo
                raw_temp = (raw_temp / 64.0) - 273.15

                pred_seg_mask, time_taken = segObj.run_inference(raw_temp)

                im1.set_array(raw_temp)
                im2.set_array(pred_seg_mask)
                # plt.imshow(raw_temp, cmap='gray')
                # plt.imshow(pred_seg_mask, cmap='seismic', alpha=0.5)
                # plt.show()

        return im1, im2


    ani = FuncAnimation(
        fig, 
        live_capture,
        init_func=init,
        cache_frame_data=False,
        interval = 1,
        blit=True
        )

    plt.show()



