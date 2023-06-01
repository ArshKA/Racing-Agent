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
### Encoder
![Encoder](media/.png)

### Decoder
![Decoder](media/.png)

  
## 3. Training
- Utilizing Google Colab's Tesla T4
<br>

![Loss Graph](media/loss_chart.png)

# Evolutionary Algorithm
Producing the best agent through genetic selection

## Agent Model
  - Impemented in vanilla NumPy for speed and flexibility
  - Can use pyTorch for larger models

![Agent Model](media/.png)



## Reporoduction values

Rank   | Mutant Offspring
| :------------: | :-------------: |
|1      |        10        |
|2      |        5         |
|3      |        4         |
|4      |        3         |
|5      |        2         |
|6      |        1         |
|Base      |        5         |
|Total Agents      |        30         |




## Training
![Max Reward Graph](media/reward_chart.png)
(Max Reward)

## Results

| <center>**0 Generations**</center> | <center>**20 Generations**</center> |
| :------------: | :-------------: |
| ![0 Generations](media/runs/test1.gif) | ![20 Generations](media/runs/test20.gif) |
| <center>**60 Generations**</center> | <center>**110 Generations**</center> |
| ![60 Generations](media/runs/test50.gif) | ![1300 Generations](media/runs/test130.gif) |



## Reflection


  

