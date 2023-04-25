import cv2
import numpy as np
import matplotlib.pyplot as plt

# Interactive plot mode
import matplotlib

matplotlib.use('TkAgg')


def show_image(image: np.ndarray, grayscale=True):
    plt.figure()
    if grayscale:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()
    return image


def show_coverage_image(images, *hide_images, show_image_idx=True, no_gui=False):
    coverage_image = np.zeros_like(images[0])
    img_indices: list[int] = []
    img_centroids: list[tuple[float, float]] = []
    for index, image in enumerate(images, start=1):
        if image is None or index in hide_images:
            continue
        _, threshold_image = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY)
        coverage_image += threshold_image
        if show_image_idx:
            gray_img = cv2.cvtColor(threshold_image, cv2.COLOR_BGR2GRAY) if len(
                threshold_image.shape) > 2 else threshold_image
            if gray_img.max() == 0:
                continue
            M = cv2.moments(gray_img)
            img_centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
            img_indices.append(index)

    try:
        if not coverage_image:
            print('No images to show')
            return
    except ValueError:
        pass

    colormap = plt.get_cmap('inferno', coverage_image.max() + 1)
    coverage_image = (colormap(coverage_image) * 2 ** 8).astype(np.uint8)[:, :, :3]
    coverage_image = cv2.cvtColor(coverage_image, cv2.COLOR_BGR2RGB)

    for idx, centroid in zip(img_indices, img_centroids):
        cX, cY = centroid
        cv2.circle(coverage_image, (cX, cY), 2, (255, 0, 0), -1)
        cv2.putText(coverage_image, str(idx), (cX - 15, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if not no_gui:
        plt.figure()
        # plt.imshow(coverage_image, cmap='hot')
        plt.imshow(coverage_image)
        plt.show()
    return coverage_image


def compare_images(image1, image2, grayscale=True):
    plt.figure()

    if grayscale:
        plt.subplot(1, 2, 1)  # last param is the subplot number
        plt.imshow(image1, cmap='gray', vmin=0, vmax=255)
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap='gray', vmin=0, vmax=255)
    else:
        plt.subplot(1, 2, 1)  # last param is the subplot number
        plt.imshow(image1, vmin=0, vmax=255)
        plt.subplot(1, 2, 2)
        plt.imshow(image2, vmin=0, vmax=255)

    plt.show()


def add_keypoints(image: np.ndarray, keypoints):
    annotated_image = np.zeros(image.shape)
    annotated_image = cv2.drawKeypoints(image, keypoints, annotated_image,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return annotated_image


def draw_matches(image1: np.ndarray, image2: np.ndarray, matcher, matches_mask=None):
    unmatched_color = 255, 0, 0
    matched_color = 0, 255, 0
    # unmatched_color = None
    # matched_color = None
    annotated = cv2.drawMatches(image1, matcher.source_data.keypoints, image2, matcher.dest_data.keypoints,
                                outImg=None, matchColor=matched_color,
                                singlePointColor=unmatched_color, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
                                # singlePointColor=unmatched_color, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                # singlePointColor=unmatched_color, flags=cv2.DrawMatchesFlags_DEFAULT,
                                matchesMask=matches_mask, matches1to2=tuple(matcher.get_matched()), matchesThickness=1)
    return annotated
