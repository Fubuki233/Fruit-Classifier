# Create an Image Classifier (CNN model) to classify images of fruits correctly

A Fruits dataset is provided that consists of these 4 classes:

- Apple only
- Orange only
- Banana only
- A mix of Fruits

Use the images in **Train.zip** and **Test.zip** to train and test your image classifier.

You are allowed to modify your training data (e.g. add/remove images), but not your testing data.

Document your experiments and results in improving your model’s accuracy.

The following activities can improve your model’s accuracy:

- Balance out the number of samples in each class
- Correct any mis-labelling in any of the 4 classes
- Determining the image-sizes to be used for training
- Image Augmentation to generate more data
- Generate any plots that you think can help the reader understand your work better.

---

## Data

The data `train.zip` and `test.zip` are found under the **Files/Team Project** menu.

---

## Remote GPU Server Setup 

**Speed Up Training with GPU Support** MacBooks/slims don't have NVIDIA GPUs for deep learning. Connect to a remote GPU server for **significantly faster training**.

### Quick Setup
0. **Sync Data to Remote Server(if needed)**
1. run:
   ```
   pip install paramiko scp tqdm
   python pic_sync.py
   ```
2. **Open CNN.ipynb in VS Code**

3. **Select Remote Kernel**:
   - Click kernel selector (top-right)
   - Select "Existing Jupyter Server"
   - Enter URL: `http://ctrl.zyh111.icu:8011/`
   - Enter password: `123456`
   - Select the `TensorFlow` kernel
4. **Verify GPU Access**:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

5. **Start Training** - Your model will now train on GPU!

---

# CA Rubics

| CRITERIA | DESCRIPTION | POINTS |
|----------|-------------|---------|
| Model Accuracy | Try to train your model to obtain **at least 92% accuracy** on the test data. | **5** |
| Model Training Experiments | Experiments done to achieve better model accuracy (e.g. pre-processing, different network architectures) | **15** |
| Report | Documentation of your training process, generation of plots and analysis of changes in results. | **5** |

---

# Deliverables

Hand in a **zip file** of your report and source codes.  
Your report should contain details of your experiments.  
In addition, provide plots to summarise your experiments.

Name your submission as `<your_team_number>.zip`.

For example, if you are in Team 1, the submitted filename should be:

