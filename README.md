# Brain_Tumor_Detection

This project leverages a deep learning algorithm, utilizing NVIDIA's Jetson-Inference framework and the ResNet-18 architecture, to detect and classify cancerous tumors in brain MRI scans. The model is trained to differentiate between healthy tissue and various types of tumors, optimized for deployment on NVIDIA Jetson devices, making it a powerful and efficient tool for early diagnosis and treatment planning in medical imaging.

![image](https://github.com/user-attachments/assets/f3c6a1ba-6275-494a-b543-79abc7792bfa)  ![image](https://github.com/user-attachments/assets/04e4ae01-fade-4557-804e-807b082dd17e)


## The Algorithm
This project utilizes the ResNet-18 architecture within the NVIDIA Jetson-Inference framework to analyze MRI images of the brain and detect tumors. 

ResNet-18 is a convolutional neural network (CNN) known for its efficiency and accuracy, particularly in image classification tasks.

1) MRI images undergo preprocessing to normalize the input data and enhance features relevant to tumor detection.
2) ResNet-18 is employed, where it efficiently learns complex features from the images while mitigating the problem of vanishing gradients through residual connections.
3) The NVIDIA Jetson-Inference framework provides a streamlined process for deploying deep learning models on edge devices, particularly NVIDIA's Jetson platform. It handles the optimization and inference processes, ensuring the model runs efficiently in real-time applications.
4) The model is trained on a labeled dataset of 4,000 total brain MRI scans, fine-tuned to optimize tumor detection. Data augmentation techniques are used to improve the model's robustness.
5) After training, the model is evaluated on a separate test set to assess its accuracy and precision. This evaluation ensures that the model generalizes unseen data and is reliable for clinical application.

## Running this project

1. Connect to Visual Studio Code using your IP address and open a new terminal
   
2. Install Jetson Inference and Docker Image https://github.com/dusty-nv/jetson-inference
   
3. Change directories to jetson-inference/python/training/classification/data and download the brain MRI scans using the command: 	`wget https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset -0 brain_tumor_detection.tar.gz tar xvzf brain_tumor_detection.tar.gz`
   
4. 	Use `cd` to navigate back into the directory **nvidia/jetson-inference** and run the following command to configure memory commitment: `echo 1 | sudo tee /proc/sys/vm/overcommit_memory`
   
5. Start the docker container using `./docker/run.sh`
    
6. Change directories to **jetson-inference/python/training/classification** from inside the docker

7. Begin training the model using the command for 50 trials `python3 train.py --model-dir=models/brain_tumor_detection data/brain_tumor_detection --epochs=50` (it may take a while to run this)

8. Run the onnx export script `python3 onnx_export.py --model-dir=models/brain_tumor_detection`

9. Your retrained model will be found in **jetson-inference/python/training/classification/models/brain_tumor_detection** under the name resnet18.onnx

10. Exit your docker using **Ctrl + D** then navigate to **jetson-inference/python/training/classification** directory

11. Use `ls models/brain_tumor_detection/` to ensure that the model is on the nano; make sure there is a file called resnet18.onnx

12. Set the NET and DATASET variables `NET=models/brain_tumor_detection` and `DATASET=data/brain_tumor_detection`

13. Test the model for "Cancer_Detected" first using `imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/Cancer_Detected/Cancer4.jpg first_test.jpg
