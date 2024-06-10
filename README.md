# PPE Detection using YOLO-NAS and Flask

## Training Model

### 1. Dataset
Dataset diimport dari [Roboflow](https://universe.roboflow.com/project-uyrxf/ppe_detection-v1x3l). Dataset ini memiliki kualitas yang baik, dari segi anotasi, variasi dan kualitas gambar yang baik.

### 2. Pemilihan Model
Model yang digunakan adalah YOLO-NAS. YOLO-NAS adalah model object detection baru dari Deci. YOLO-NAS memiliki performa yang baik dibanding model yolo yang lain dari segi kualitas dan efektivitas.
![YOLO_NAS](https://github.com/bayudaru2020/PPE-Detection-using-YOLO-NAS-and-Flask/blob/master/img/yolo_nas_peforma.png)

### 3. Hasil Deteksi
Image detection
<p align="center">
  <img src="https://github.com/bayudaru2020/cctv_region_counter/blob/Bayu-Daru-Isnandar-branch/result-example-img.png" width="50%">
</p>

Video detection
![result-vid](https://github.com/bayudaru2020/cctv_region_counter/blob/Bayu-Daru-Isnandar-branch/img/result-example-vid-gif.gif)



## Inference with Simple flask Web
### Step 1: Install flask and the Required Libraries
Create dependencies and install required libraries
* python==3.10.0
* flask==3.0.3
* ultralytics==8.2.14
* super-gradients

### Step 2: clone this repo & download my model
```bash
# Clone this repo
git clone https://github.com/bayudaru2020/cctv_region_counter.git
```
- Setelah melakukan clone, buka file link-download-finetuned-model.txt
- download model yang sudah dilatih
- letakan model pada folder web-app-flask/model/

### Step 3 Jalankan flask
* Jalankan file web.py pada path web-app-flask/web.py (Pastikan menjalankan pada root dan interpreter yang sudah dibuat sebelumnya)
* Buka port http://192.168.0.171:5000 menggunakan browser

Simple web flask siap digunakan
![simple-web-flask](https://github.com/bayudaru2020/cctv_region_counter/blob/Bayu-Daru-Isnandar-branch/img/simple-web-flask.png)

Selain menggunakan simple-web-flask, model dapat diinference langusng dari program python tanpa interface pada file Inference_Yolo_Nas_with_Python.py
