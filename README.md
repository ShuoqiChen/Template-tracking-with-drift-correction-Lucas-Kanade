# Template tracking with drift correction - Lucas-Kanade


**What is Lucas-Kanade Method?*

In computer vision, the Lucas–Kanade method is a widely used method for object tracking by Bruce D. Lucas and Takeo Kanade. 

The algorithm that the optical flow is essentially constant in a local neighbourhood of the pixel under consideration, and solves the basic optical flow equations for all the pixels in that neighbourhood, by the least squares criterion.
The benefit of Lucas–Kanade method can often resolve the inherent ambiguity of the optical flow equation. It is also less sensitive to image noise than point-wise methods. 

**What is in the demo?*

In this demo, I demonstrate the robustness and versatility of Lucas-Kanade method by showing the the tracker successful handle different object tracking in video. 
1) Car moving on the street
2) Toy manipulated by a moving hand 
3) Convoy of cars driving in an open space

1. Car moving on the street
1. Toy manipulated by a moving hand 
1. Convoy of cars driving in an open space

In the demo vidoes, I also use template correction method proposed by Iain Matthews et al. (2003, https://www.ri.cmu.edu/publication_view.html?pub_id=4433) to solve the template drifting problem. 

![demo](demo/car_with_drift_correction.gif)

![demo](demo/sylv_with_drift_correction.gif)

![demo](demo/airfield_forward_k_motion_tracking.gif)
