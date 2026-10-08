-------------------
GENERAL INFORMATION
-------------------

Title and description of the dataset:

YOLO-based models trained via transfer learning with top-view person image annotations. 

YOLO models customized for detecting and classifying the person class in top-down (zenithal) views. 
The original YOLO architectures were adapted by disabling the detection of frontal or lateral human views and removing all other object classes recognized by the base models. The training dataset consists of 3,137 manually annotated images collected from three university libraries, capturing a rich variety of top-view human patterns, including individuals alone, overlapping people, partial occlusions, and diverse clothing styles.

Transfer learning was applied to YOLO versions v5, v8, and v11, using both their nano variants (optimized for low computational cost) and xlarge variants (providing higher accuracy at the expense of greater computational demand).

It also provides a Python script (.py) to run each model on sample images for person detection.
These sample images were not used during the training or validation phases.

Author's contact information (name, email, ORCID, institution)

Lucio Hernando Cánovas
lucio.hernando@edu.upct.es
https://orcid.org/0009-0006-4463-6133
Universidad Politécnica de Cartagena

Alejandro S. Martínez-Sala
alejandros.martinez@upct.es
https://orcid.org/0000-0002-2825-8965
Universidad Politécnica de Cartagena

Juan Carlos Aarnoutse-Sánchez
juanc.sanchez@upct.es
https://orcid.org/0000-0002-5330-1160
Universidad Politécnica de Cartagena

Juan Jose Álcaráz Espín
juan.alcaraz@upct.es
https://orcid.org/0000-0002-1756-0130
Universidad Politécnica de Cartagena

----------------------------------
SHARING/ACCESS/CONTEXT INFORMATION
----------------------------------

Was data derived from another source? If so, please add link where such work is located:

The YOLO-based models included in this dataset are fine-tuned versions 
derived from the official Ultralytics YOLO implementations. 

Citation:
JOCHER, Glenn y QIU, Jing. Ultralytics YOLO11 [programa informático]. 
Versión 11.0.0, 2024. Disponible en: https://github.com/ultralytics/ultralytics 
[Acceso: 2025-06-21]. Licencia: AGPL-3.0.

--------------------
DATA & FILE OVERVIEW
--------------------

File List (if file is compressed, list its content): 
/images → A test set of .jpg images depicting people from a top-view camera perspective.

models: → a set of YOLO models in .pt format, including versions 5, 8, and 11  ( _n nano,  _x xtralarge)
yolov5_n.pt

yolov8_n.pt

yolov11_n.pt

yolov5_x.pt

yolov8_x.pt

yolov11_x.pt

Yolo_test.py → A Python script demonstrating how to use a YOLO model to detect people in the sample images. You can edit the script to select the model and specify the paths to the image folder and the model files.

File format (provide a list of all file formats present in this dataset):
.jpg → image files

.pt → YOLO model files

.py → Python script file