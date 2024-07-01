# Pencil Eraser Detection on Yolov5 

## Aim And Objectives 
Aim 

The aim of this project is to develop and train a YOLOv5 model capable of accurately detecting pencil erasers in images and videos. The objective is to leverage YOLOv5's state-of-the-art object detection capabilities to create a reliable tool for identifying pencil erasers in various settings.

objectives 

  1. Data Collection and Annotation

 - Collect a diverse dataset of images containing pencil erasers.
- Annotate the images with bounding boxes to indicate the presence and location of pencil erasers.

2. Model Training

- Configure and set up the YOLOv5 environment.
- Train the YOLOv5 model on the annotated dataset.
- Optimize the model by adjusting hyperparameters to achieve high accuracy.

3. Model Evaluation

- Evaluate the trained model on a validation dataset to assess its performance.
- Use metrics such as precision, recall, and mAP (mean Average Precision) to measure accuracy.

4. Inference

- Implement the trained model for inference on new images and videos.
- Demonstrate real-time detection capabilities of the model.

5. Integration and Deployment

- Provide a guide for integrating the model into applications using Python.
- Ensure the model can be deployed in various environments, including local systems and cloud platforms.

6. Documentation and Support

- Create comprehensive documentation to guide users through setup, training, and usage of the model.
- Offer troubleshooting tips and contact information for user support.

## Abstract

This project focuses on the development of a YOLOv5 model for detecting pencil erasers in images and videos. YOLOv5, a leading object detection algorithm, is renowned for its speed and accuracy, making it ideal for real-time applications. The goal is to create a reliable model that can identify pencil erasers in various environments, which can be utilized in educational tools, inventory management, and other applications requiring object detection. This README outlines the aim, objectives, setup process, training procedure, and evaluation metrics for the project.

## Introduction

In the modern digital age, object detection has become a critical component in various applications ranging from security systems to inventory management and beyond. YOLOv5 (You Only Look Once version 5) is one of the most advanced and efficient object detection algorithms available today, known for its speed and accuracy. This project focuses on harnessing the power of YOLOv5 to detect pencil erasers in images and videos, which can have practical applications in educational tools, office supply management, and automated retail systems.

The detection of pencil erasers may seem straightforward, but it involves several challenges such as varying lighting conditions, different backgrounds, and the presence of multiple objects in the same image. This project aims to overcome these challenges by training a YOLOv5 model on a diverse and well-annotated dataset of pencil erasers, thereby creating a robust tool capable of accurate detection in real-world scenarios.

The following sections of this README will guide you through the aims and objectives of this project, the setup process, the steps for training the model, and how to use the trained model for inference. By the end of this document, you will have a comprehensive understanding of how to create, train, and deploy a YOLOv5 model for pencil eraser detection.

## Literature Review

This literature review provides a comprehensive overview of the evolution of object detection methods, focusing on YOLOv5 and its applications, particularly in detecting pencil erasers. Adjustments and additions can be made to align with specific project requirements and research findings.

## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

## Proposed System

The proposed system leverages YOLOv5 (You Only Look Once version 5) for the detection of pencil erasers in images and videos. YOLOv5 is chosen due to its balance between accuracy and speed, making it suitable for real-time applications. This system aims to build a robust and efficient pipeline for detecting pencil erasers, which includes dataset collection, model training, evaluation, and deployment.

## Methodology

Traditional Methods
Early object detection techniques relied heavily on hand-crafted features and traditional machine learning algorithms. Notable methods include:

- Histogram of Oriented Gradients (HOG): Introduced by Dalal and Triggs (2005), HOG is a feature descriptor used to detect objects, primarily humans. It works by calculating gradient orientations in a dense grid of uniformly spaced cells and using overlapping local contrast normalization.

- Deformable Part Models (DPM): Proposed by Felzenszwalb et al. (2008), DPM uses a set of parts and a deformable spatial model to capture the variability in object shapes. It employs a sliding window approach for detection, which is computationally expensive.
vim ~/.bashrc


## Installation

#### Initial Configuration

sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*

#### Create Swap

udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
##### make entry in fstab file
/swapfile1	swap	swap	defaults	0 0 

#### Cuda env in bashrc

vim ~/.bashrc

##### add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

#### Update & Upgrade

sudo apt-get update

sudo apt-get upgrade

#### Install some required Packages

sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev

sudo pip3 install pillow

#### Install Torch

curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"

#### Install Torchvision

git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install

#### Clone Yolov5

git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt

#### Download weights and Test Yolov5 Installation on USB webcam

sudo python3 detect.py

sudo python3 detect.py --weights yolov5s.pt  --source 0

## Advantages

1. Clarity and Organization

Project Overview: Provides a clear and concise summary of the project's purpose, objectives, and scope.
Structure: Organizes the project's details in a structured format, making it easier for others to understand and follow.

2. Ease of Use

Instructions: Offers step-by-step instructions for setting up the environment, running the code, and using the trained model, which is crucial for both novice and experienced users.
Dependencies: Lists all the required dependencies and installation commands, ensuring users can set up the project without issues.

3. Documentation

Methodology: Documents the entire methodology, including data collection, preprocessing, model training, evaluation, and deployment.
Configuration: Provides details on the configuration files and parameters used in the YOLOv5 model.

4. Reproducibility

Consistency: Ensures that anyone who wants to replicate the project can do so accurately by following the documented steps.
Version Control: Details the versions of libraries and tools used, reducing compatibility issues.

5. Collaboration

Teamwork: Facilitates collaboration among team members by providing a common reference point.
Community Contribution: Encourages contributions from the wider community by making it easy for others to understand and improve the project.

## Application 

Detecting pencil erasers using YOLOv5 has several practical applications across various domains. Here are some key applications:

1. Educational Tools and Applications

Interactive Learning: In educational software, detecting pencil erasers can help create interactive learning tools where students can learn about different stationery items.
Smart Classroom Management: Detecting pencil erasers and other stationery items can assist teachers in managing classroom supplies and ensuring that students have all necessary materials.

2. Inventory Management

Stationery Stores: Automated systems in stationery stores can use object detection to manage inventory, track stock levels, and identify misplaced items.
Warehouse Management: In warehouses, detecting pencil erasers can help in sorting and organizing inventory efficiently.

3. Retail Automation

Self-Checkout Systems: Integrating pencil eraser detection into self-checkout systems in retail stores can help automate the checkout process and reduce errors.
Product Placement Analysis: Retailers can analyze product placement and optimize shelf organization based on detection data.

4. Manufacturing and Quality Control

Production Line Monitoring: In pencil eraser manufacturing, object detection can monitor the production line for quality control, ensuring that products meet the required standards.
Defect Detection: Detecting defects or irregularities in pencil erasers during the manufacturing process can improve product quality and reduce waste.

5. Robotics and Automation

Pick and Place Robots: Robots equipped with YOLOv5 detection capabilities can identify and handle pencil erasers, aiding in tasks such as sorting, packing, and assembling products.
Service Robots: Service robots in educational institutions or offices can use object detection to assist in organizing and managing stationery supplies.

## Future Scope

1. Improved Model Accuracy

Advanced Architectures: Explore and integrate more advanced YOLO architectures (e.g., YOLOv6, YOLOv7) to enhance detection accuracy and efficiency.
Transfer Learning: Utilize transfer learning from models pre-trained on larger, more diverse datasets to improve performance on pencil eraser detection.

2. Expanded Dataset

Diverse Data Collection: Collect more diverse and extensive datasets with various backgrounds, lighting conditions, and eraser types to improve model robustness.
Synthetic Data Generation: Use synthetic data generation techniques to create more training samples, especially for rare or less common scenarios.

## Conclusion

The proposed methodology provides a structured approach to developing a YOLOv5-based system for detecting pencil erasers. This includes comprehensive steps from data collection and annotation to model training, evaluation, and deployment. By leveraging the capabilities of YOLOv5, the system aims to achieve high accuracy and real-time detection performance, making it applicable to various practical scenarios. Future improvements could include incorporating more advanced YOLO versions and exploring other deep learning techniques to enhance detection accuracy.

## Reference
1] Roboflow:- https://roboflow.com/

2] Datasets or images used :- https://www.gettyimages.ae/search/2/image?phrase=helmet

3] Google images

## Articles :-
1] https://www.bajajallianz.com/blog/motor-insurance-articles/what-is-the-importance-of-wearing-a-helmet-while-riding-your-two-wheeler.html#:~:text=Helmet%20is%20effective%20in%20reducing,are%20not%20wearing%20a%20helmet.

2] https://www.findlaw.com/injury/car-accidents/helmet-laws-and-motorcycle-accident-cases.html








