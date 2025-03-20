This brain tumor dataset by Cheng, Jun, et al (https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) contains 3064 T1-weighted contrast-inhanced images from 233 patients with three kinds of brain tumor: meningioma (708 slices), glioma (1426 slices), and pituitary tumor (930 slices).

1. Data structure
The folders in this directory are named after the labels of the data in them. For each datapoint, the .jpg file is the image, and the corresponding .txt file stores the information of [label, x_center, y_center, box_width, box_height].
The labels are numbered as: 1 for meningioma, 2 for glioma, 0 for pituitary tumor.

2. The acquisition protocol used to acquire the images
All images in the dataset (T1-weighted contrast-enhanced MRI) were acquired after Gd-DTPA injection at Nanfang Hospital, Guangzhou, China and General Hospital, Tianjin Medical University, China from 2005.9 to 2010.10. The images have an in-plane resolution of 512 × 512 with pixel dimensions of 0.49 × 0.49 mm^2 . The slice thickness is 6 mm and the slice gap is 1 mm. The Gd dose was 0.1 mmol/kg at a rate of 2 ml/s.

No further details are available due to the long passage of time. 

3. Citations
This data was used in the following papers:
1. Cheng, Jun, et al. "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition." PloS one 10.10 (2015).
2. Cheng, Jun, et al. "Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation." PloS one 11.6 (2016). Matlab source codes are available on github https://github.com/chengjun583/brainTumorRetrieval

4. Contact information
Jun Cheng
School of Biomedical Engineering
Shenzhen University, Shenzhen, China
Email: chengjun583@qq.com

