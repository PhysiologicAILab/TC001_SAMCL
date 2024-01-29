import platform
import subprocess
import time
from typing import Tuple
import cv2
import numpy as np


class VideoStreamFFmpeg:
    def __init__(self, src: int, fps: int, resolution: Tuple[int, int]):
        self.src = src
        self.fps = fps
        self.resolution = resolution
        self.pipe = self._open_ffmpeg()
        self.frame_shape = (self.resolution[1], self.resolution[0], 3)
        self.frame_size = np.prod(self.frame_shape)
        self.wait_for_cam()

    def _open_ffmpeg(self):
        os_name = platform.system()
        if os_name == "Darwin":  # macOS
            input_format = "avfoundation"
            video_device = f"{self.src}:none"
        elif os_name == "Linux":
            input_format = "v4l2"
            video_device = f"{self.src}"
        elif os_name == "Windows":
            input_format = "dshow"
            video_device = f"video={self.src}"
        else:
            raise ValueError("Unsupported OS")

        command = [
            'ffmpeg',
            '-f', input_format,
            '-r', str(self.fps),
            '-video_size', f'{self.resolution[0]}x{self.resolution[1]}',
            '-i', video_device,
            '-vcodec', 'mjpeg',  # Input codec set to mjpeg
            '-an', '-vcodec', 'rawvideo',  # Decode the MJPEG stream to raw video
            '-pix_fmt', 'bgr24',
            '-vsync', '2',
            '-f', 'image2pipe', '-'
        ]

        if os_name == "Linux":
            command.insert(2, "-input_format")
            command.insert(3, "mjpeg")

        return subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
        )

    def read(self):
        raw_image = self.pipe.stdout.read(self.frame_size)
        if len(raw_image) != self.frame_size:
            return None
        image = np.frombuffer(
            raw_image, dtype=np.uint8).reshape(self.frame_shape)
        return image

    def release(self):
        self.pipe.terminate()

    def wait_for_cam(self):
        for _ in range(30):
            frame = self.read()
        if frame is not None:
            return True
        return False


class VideoStreamCV:
    def __init__(self, src: int, fps: int, resolution: Tuple[int, int]):
        self.src = src
        self.fps = fps
        self.resolution = resolution
        self.cap = self._open_camera()
        self.wait_for_cam()

    def _open_camera(self):
        cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()

    def wait_for_cam(self):
        for _ in range(30):
            frame = self.read()
        if frame is not None:
            return True
        return False


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"Main function time: {round(t1-t0, 4)}s")
        return result

    return wrapper


def computation_task():
    for _ in range(5000000):
        9999 * 9999


@timeit
def run(cam: VideoStreamCV | VideoStreamFFmpeg, run_task: bool):
    timer = []
    for _ in range(100):
        t0 = time.perf_counter()
        cam.read()
        timer.append(time.perf_counter() - t0)

        if run_task:
            computation_task()

    cam.release()
    return round(np.mean(timer), 4)


def main():
    fsp = 25
    # resolution = (1920, 1080)
    resolution = (256, 192)

    for run_task in [False, True]:
        ff_cam = VideoStreamFFmpeg(src=0, fps=fsp, resolution=resolution)
        cv_cam = VideoStreamCV(src=0, fps=fsp, resolution=resolution)

        print(f"FFMPEG, task {run_task}:")
        print(f"Mean frame read time: {run(cam=ff_cam, run_task=run_task)}s\n")
        print(f"CV2, task {run_task}:")
        print(f"Mean frame read time: {run(cam=cv_cam, run_task=run_task)}s\n")


if __name__ == "__main__":
    main()
