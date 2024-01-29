import cv2
import numpy as np
import argparse
from inference import ThermSeg
from utils.configer import Configer
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import sys
from utils.sig_process import lFilter
import threading

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

global resp_sig, capture
# capture = True

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

    width = 256  # Sensor width
    height = 192  # sensor height
    nCh = 1 #3 #1
    fps = 25

    if 'linux' in sys.platform.lower():
        cap = cv2.VideoCapture('/dev/video'+str(args_parser.device), cv2.CAP_V4L)
    elif 'win' in  sys.platform.lower():
        cap = cv2.VideoCapture(int(args_parser.device), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(int(args_parser.device))
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # cap.set(10, 150)
    cap.set(cv2.CAP_PROP_FPS, fps)

    configer = Configer(args_parser=args_parser)
    ckpt_root = configer.get('checkpoints', 'checkpoints_dir')
    ckpt_name = configer.get('checkpoints', 'checkpoints_name')
    configer.update(['network', 'resume'], os.path.join(ckpt_root, ckpt_name + '.pth'))
    segObj = ThermSeg(configer)
    segObj.load_model(configer)

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(1, 2, figsize=(16,8))

    init_image1 = (np.random.rand(height, width) + 1) * 15
    im1 = ax[0].imshow(init_image1, cmap='gray', vmin=0, vmax=40)

    init_image2 = (np.random.rand(height, width) + 1) * 3
    im2 = ax[0].imshow(init_image2, cmap='seismic', alpha=0.5, vmin=0, vmax=6)

    nSeconds = 10
    resp_sig = np.zeros(fps*nSeconds)
    x_axis = np.arange(0, nSeconds, 1/fps)
    line1 = Line2D([], [], color='blue')
    ax[1].add_line(line1)
    # ln = ax[1].plot(resp_sig)

    ax[1].set_xlim(0, nSeconds)
    # ax[1].autoscale(enable=True, axis='y', tight=True)
    ax[1].set_ylim(-1, 1)

    extract_breathing_signal = True
    resp_lowcut = 0.1
    resp_highcut = 0.5
    filt_order = 2
    resp_filt_obj = lFilter(resp_lowcut, resp_highcut, fps, order=filt_order)

    # initialiaze filter
    for rsp in resp_sig:
        rsp_val = resp_filt_obj.lfilt(25)


    # def capture_n_process_thread():
    #     global capture, resp_sig


    def init():
        global resp_sig
        ax[0].set_axis_off()
        line1.set_data(x_axis, resp_sig)
        return im1, im2, line1


    def update_fig(i):
        global resp_sig

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

                # print(raw_temp.shape)
                # exit()

                min_temp = np.round(np.min(raw_temp), 2)
                max_temp = np.round(np.max(raw_temp), 2)

                pred_seg_mask, time_taken = segObj.run_inference(raw_temp)
                # 0 - background, 1 - chin, 2 - mouth, 3 - eyes, 4 - eyebrows, 5 - nose
                nose_label = 5

                im1.set_array(raw_temp)
                im2.set_array(pred_seg_mask)

                if extract_breathing_signal:
                    respVal = 0
                    bbox_corners = np.argwhere(pred_seg_mask == nose_label)
                    if bbox_corners.size > 0:
                        nose_pix_min_y, nose_pix_min_x = bbox_corners.min(0)
                        nose_pix_max_y, nose_pix_max_x = bbox_corners.max(0)
                        nostril_box = raw_temp[nose_pix_max_y-10:nose_pix_max_y, nose_pix_min_x:nose_pix_max_x]
                        nostril_box_label = pred_seg_mask[nose_pix_max_y-10:nose_pix_max_y, nose_pix_min_x:nose_pix_max_x]

                        nostril_seg_matrix = nostril_box[nostril_box_label == nose_label]
                        try:
                            respVal = np.mean(nostril_seg_matrix)
                        except:
                            # print('Missed RoI')
                            pass

                    else:
                        nose_mask = raw_temp[pred_seg_mask == nose_label]
                        if nose_mask.size > 0:
                            try:
                                respVal = np.mean(nose_mask)
                            except:
                                # print('Missed RoI')
                                pass
                            info_str = info_str + "; Nostril extraction failed, using whole nose mask"
                        else:
                            respVal = max_temp
                            info_str = info_str + "; Nose not detected!!"

                    resp_sig = np.roll(resp_sig, -1)
                    filt_respVal = resp_filt_obj.lfilt(respVal)

                    resp_sig[-1] = filt_respVal
                    # resp_sig[-1] = respVal

                    min_resp = np.min(resp_sig)
                    max_resp = np.max(resp_sig)
                    ax[1].set_ylim(min_resp, max_resp)

                    line1.set_data(x_axis, resp_sig)

                # plt.imshow(raw_temp, cmap='gray')
                # plt.imshow(pred_seg_mask, cmap='seismic', alpha=0.5)
                # plt.show()

        return im1, im2, line1


    ani = FuncAnimation(
        fig, 
        update_fig,
        init_func=init,
        cache_frame_data=False,
        interval = 40,
        blit=True
        )

    plt.show()



# '''
