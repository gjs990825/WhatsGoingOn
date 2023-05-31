import cv2


def get_offsets(all_frames, extract_frames, start=0, end=-1):
    if end == -1:
        end = all_frames - 1
    if not 0 <= start <= end < all_frames:
        return []

    frame_count = end - start + 1
    if not extract_frames <= frame_count:
        return []

    avg_len = frame_count / extract_frames

    return [int(start + (i + 0.5) * avg_len) for i in range(extract_frames)]


def video_frame_extract(extract_frames, video_path, start=0, end=-1):
    capture = cv2.VideoCapture(video_path)
    all_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    offsets = get_offsets(all_frames, extract_frames, start, end)
    assert len(offsets) > 0

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    images = []
    index = start
    for offset in offsets:
        while index <= offset:
            if not capture.grab():
                raise RuntimeError('Video/Camera grabbing error')
            index += 1
        success, img = capture.retrieve()
        index += 1
        if not success:
            raise RuntimeError('Video/Camera retrieving error')

        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return images
