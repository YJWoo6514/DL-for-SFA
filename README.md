Project Title:
Deep Learning–Based Detection of Peripheral Arterial Lesions Using Faster R-CNN on Lower Extremity MIP CT Images

Overview:
This repository contains the source code and training pipeline used in our pilot study investigating the feasibility of identifying peripheral arterial lesions using the Faster R-CNN object detection algorithm. The model was trained on Maximum Intensity Projection (MIP) images reconstructed from contrast-enhanced lower extremity CTA scans. Our goal was to evaluate whether algorithm-based detection could reliably identify lesions such as stenosis, occlusion, and stents within superficial femoral arteries (SFA), thereby supporting the future development of clinically applicable decision support systems (CDSS).
All training was performed using default Faster R-CNN hyperparameters, with performance evaluated based on objectness loss, classification loss, and mAP@0.5
