# Genetic-CarRacing-v2
Solving Gym's car racing environment utilizing genetic algorithms, neural networks, and auto-encoders.

# Encoding
Compressing the image observation into a 1D vector

## 1. Data Collection
- Random actions
- Save every nth image
  ### Image Procesing
  - Convert to grayscale
  - Remove gridding patterns
  <br>

  | Original Image | Processed Image |
  | :----------------: | :----------------: |
  | ![Original Image 1](media/original1.png) | ![Processed Image 1](media/processed1.png) |
  | ![Original Image 2](media/original2.png) | ![Processed Image 2](media/processed2.png) |




  ### Stacking Frames
  - Stack 3 consecutive frames
  - Determine velocity & acceleration
  <br>
  | Slow | Fast |
  | :----------------: | :----------------: |
  | ![Slow Stacked](media/slow_stacked.png) <br>Thin outline indicates little difference <br> between frames| ![Fast Stacked](media/fast_stacked.png) <br>Wide outline indicates larger difference, <br> meaning the car's speed is greater|

## 2. Model Architecture
- Compresses 96x96x3 image input into 32 values
- 99.8843% reduction
  Diagram
  
## 3. Training
- On Colab Tesla T4
Loss Diagram

# Evolutionary Algorithm
Producing the best agent through genetic selection

## Reporoduction values

## Model architecture

## Reward Graph

## Reflection


  

