# Autonomous Rock Detection for Asteroid Missions Using Deep Learning and Synthetic Data

This thesis work authored by Taylor Jensen at Northwestern University addresses the creation and evaluation of deep learning architectures which are trained only on synthetic images. The goal of these models is to autonomously detect rocks on the surface of asteroids.

![Link to Paper](https://doi.org/10.21985/n2-9xcr-6h31)

# License

Note that this work is licensed for no commercial use under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) lisence. If there are multiple license terms that are associated to the thesis paper resulting from this work and the code of this work, the more restrictive license terms shall apply.

# Abstract

Asteroids can contain secrets about the origin of the universe and have future potential commercial applications including mining, space stations, and commerce. The asteroid Bennu was selected by NASA in 2016 because it may provide information on the origins of our solar system and increase understanding asteroids that could impact earth. However, landing a probe on an asteroid can require thousands of hours of work to identify safe areas. 

Historically, this image processing task is difficult to accomplish because of the lack of images annotated with rocks, the diverse nature of asteroid surfaces, and the inclusion of human interpretation bias. These reasons restricted earlier solution development. This work contributes to existing research through the creation and evaluation of five deep learning architectures that are designed automate this task. 

The Unet architecture, in addition to four custom architectures, detect rocks using synthetic lunar images. Two of the architectures are an adaptation of Unet structure. The remaining two utilize both a grayscale image and an image processed by a Sobel edge detector. After preprocessing, the two best models are tested on real images from NASA’s mission to the asteroid Bennu and create a global rock mosaic. The study finds that Unet performs the best on synthetic data but is sensitive to the difference between the datasets and does not have the same performance on real images. 

In contrast, a custom model architecture named “Y Model, High Granularity” has the opposite results of Unet with poorer performance on synthetic images but better performance on real images. While the findings promise that there is opportunity to automate rock-finding tasks in missions, additional architectures should be explored. To that end, it is recommended that Y Model, High Granularity be used over Unet, given the improved performance.



__Example model performance on real life asteroid surface:__

![Alt Text](screenshots/Screenshot%202023-06-25%20at%205.38.53%20PM.png)