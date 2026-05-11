# Camera Motion Reconstruction

This repository contains my computer vision project on camera motion reconstruction. The goal of the project was to reconstruct a realistic camera motion path using rotation and translation estimates from an image sequence.

The project focuses on how a camera moves through a scene over time. After obtaining the rotation and translation parameters, I used those values to map the camera's path and create a visual representation of its movement.

## Project Overview

Camera motion reconstruction is an important computer vision problem because it helps estimate how a camera moves through space based on visual input. This type of work is connected to areas like visual odometry, structure from motion, robotics, autonomous systems, and 3D reconstruction.

In this project, I used the rotation and translation parameters to reconstruct the camera's trajectory. By accumulating the translation values over time, I was able to map the camera's estimated movement through space. The final result successfully portrays a realistic camera motion path instead of just showing isolated frame-to-frame movement values.

This project helped me understand how mathematical outputs from a computer vision workflow can be turned into something more interpretable. Instead of only looking at rotation and translation parameters as numbers, the final visualization shows the camera's movement as a reconstructed path.

## Main Features

- Camera motion path reconstruction
- Rotation and translation parameter processing
- Frame-to-frame movement accumulation
- 3D trajectory visualization
- Motion path interpretation
- Computer vision workflow development

## Tools and Technologies

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Methodology

The project follows a computer vision workflow for reconstructing camera movement from image-based motion estimates.

First, the rotation and translation parameters were obtained from the image sequence. These parameters describe how the camera's position and orientation changed between frames.

Then, the translation values were accumulated over time. This made it possible to reconstruct the camera's estimated trajectory instead of only analyzing each frame transition separately.

Finally, the reconstructed path was visualized. This helped make the camera motion easier to understand because the final output shows the movement as a spatial path rather than only as numerical values.

## Results

The camera motion path was reconstructed successfully. The final visualization portrays a realistic trajectory and shows that the rotation and translation parameters can be used to estimate the camera's movement over time.

The reconstructed path gives a clear visual representation of the camera's motion. It shows how individual translation estimates can be accumulated into a larger movement pattern and interpreted as a camera trajectory.
