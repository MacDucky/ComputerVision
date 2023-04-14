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
