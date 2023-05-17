# Computer Vision - an assignment based course

___
**Lecturer:** Simon Korman

**Assignees:**

Daniel Poroshan - https://github.com/MacDucky

Nir Segal - https://github.com/NirSeg
___

<!-- TOC -->
* [Computer Vision - an assignment based course](#computer-vision---an-assignment-based-course)
  * [Assignment 1](#assignment-1)
    * [Puzzle solving (via image stitching)](#puzzle-solving-via-image-stitching)
    * [Objective:](#objective)
      * [SIFT](#sift)
      * [RANSAC](#ransac)
  * [Assignment 2](#assignment-2)
<!-- TOC -->
___

## Assignment 1

### Puzzle solving (via image stitching)

### Objective:
___
Given:

* N square puzzle pieces
* An affine or homography transformation for the first puzzle piece
* Final image dimensions of the solved puzzle

Figure out a way to transform and stitch together all the puzzle pieces seamlessly into one single piece.

Input example:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img height="200" src="puzzles/puzzle_affine_1/pieces/piece_1.jpg" title="Image1" width="200"/> <img height="200" src="puzzles/puzzle_affine_1/pieces/piece_2.jpg" title="Image 2" width="200"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Affine Transformation for the first piece:

|                       |                   |                  |
|-----------------------|-------------------|------------------|
| 0.072873879065059     | -1.33125760466063 | 558.492032374654 |
| 1.46728322166917      | 0.138337505192396 | 5.04480717894527 |
| -3.13063377305189e-18 | 0                 | 1                |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_note that the last row is approximately [0,0,1] therefore this is trully an affine transformation_

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Final image size: `521x760`
___


To achieve this goal, we've utilized 2 main concepts:

* Keypoint matching in the image with [SIFT](#sift) algorithm
* [RANSAC](#ransac) image fitting and refitting


#### SIFT
#### RANSAC


___

## Assignment 2