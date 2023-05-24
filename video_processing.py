import cv2
import numpy as np


def video_frame_extract(n, video_path, start=0, end=-1):
    capture = cv2.VideoCapture(video_path)
    all_frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if end == -1:
        end = all_frame_count - 1
    assert 0 <= start <= end < all_frame_count

    frame_count = end - start + 1
    assert n <= frame_count

    average_length = int(frame_count / n)

    offsets = [int(start + average_length / 2 + i * average_length) for i in range(n - 1, -1, -1)]

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)

    index = start
    images = []
    while offsets:
        offset = offsets.pop()
        while index < offset:
            capture.grab()
            # grab() does not process frame data, for performance improvement
            index += 1

        success, img = capture.read()
        index += 1

        if not success:
            raise RuntimeError()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    return np.concatenate(images, axis=2)
