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
  Image of processing

  ### Stacking Frames
  - Stack 3 consecutive frames
  - Determine velocity & acceleration
  Image of stacking
  
## 2. Model Architecture
= Compresses 96x96x3 image input into 32 values
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


  

