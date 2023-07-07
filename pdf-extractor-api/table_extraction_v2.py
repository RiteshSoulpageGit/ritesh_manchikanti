import io
import json
import os
import warnings

import cv2
# import pypdfium2 as pdfium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import pytesseract
import pytorch_lightning as pl
import requests
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from imutils import resize
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image
from PIL import JpegImagePlugin as jplugin
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import (Concatenate, Conv2D, Conv2DTranspose,
                                     Dropout, Input, Layer, UpSampling2D)
from tensorflow.keras.models import Model
from torchvision import models, transforms
from werkzeug.utils import secure_filename

labels = 0
warnings.filterwarnings("ignore")
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


### classification architecture########
# global Classificationmodel
class Classificationmodel(pl.LightningModule):
    """Core model where it takes resnet18 by default
    creates a training step, validation step and
    test step. it also has configure optimizers"""

    def __init__(self, classes):
        super(Classificationmodel, self).__init__()
        self.save_hyperparameters()

        self.model = models.resnet18(pretrained=True)
        self.classes = classes
        self.modelname = "resnet"
        self.model = self.build_model()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 0.001

    def build_model(self):
        """returns the model based on the model name along
        with number of classes of dataset as a final layer
        in model"""
        if self.modelname in ["resnet", "inception", "googlenet"]:
            infeatures = self.model.fc.in_features
            self.model.fc = nn.Linear(infeatures, self.classes)
        if self.modelname in ["vggnet", "alexnet"]:
            infeatures = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(infeatures, self.classes)
        if self.modelname in ["mobilenet"]:
            infeatures = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(infeatures, self.classes)
        if self.modelname in ["densenet"]:
            infeatures = self.model.classifier.in_features
            self.model.classifier = nn.Linear(infeatures, self.classes)
        return self.model

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        accuracy = self.accuracy(output, y)
        self.log("train_acc_step", accuracy)
        self.log("train_loss", loss)
        print("train_loss", loss)
        print("train accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        accuracy = self.accuracy(output, y)
        self.log("val_acc_step", accuracy)
        self.log("val_loss", loss)
        # print("val_loss", loss)
        # print("val accuracy", accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        self.accuracy(output, y)
        self.log("test_acc_step", self.accuracy)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # if self.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # if self.optim.lower() == 'adam':
        # optimizer = torch.optim.Adam(self.parameters(),lr = self.lr)
        return optimizer


class TableDetectionInImage:
    def __init__(self):
        self.path = "model_15.h5"
        self.table_detect_model()

    ### model architecture for the table detection
    def table_detect_model(self):
        tf.keras.backend.clear_session()

        class table_mask(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.conv_7 = Conv2D(
                    kernel_size=(1, 1),
                    filters=128,
                    kernel_regularizer=tf.keras.regularizers.l2(0.002),
                )
                self.upsample_pool4 = UpSampling2D(
                    size=(2, 2), interpolation="bilinear"
                )
                self.upsample_pool3 = UpSampling2D(
                    size=(2, 2), interpolation="bilinear"
                )
                self.upsample_final = Conv2DTranspose(
                    filters=2,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="softmax",
                )

            def call(self, input, pool3, pool4):
                x = self.conv_7(input)
                x = self.upsample_pool4(x)
                x = Concatenate()([x, pool4])

                x = self.upsample_pool3(x)
                x = Concatenate()([x, pool3])

                x = UpSampling2D((2, 2))(x)
                x = UpSampling2D((2, 2))(x)

                x = self.upsample_final(x)
                return x

        class col_mask(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.conv_7 = Conv2D(
                    kernel_size=(1, 1),
                    filters=128,
                    kernel_regularizer=tf.keras.regularizers.l2(0.004),
                    kernel_initializer="he_normal",
                )
                self.drop = Dropout(0.8)
                self.conv_8 = Conv2D(
                    kernel_size=(1, 1),
                    filters=128,
                    kernel_regularizer=tf.keras.regularizers.l2(0.004),
                    kernel_initializer="he_normal",
                )
                self.upsample_pool4 = UpSampling2D(
                    size=(2, 2), interpolation="bilinear"
                )
                self.upsample_pool3 = UpSampling2D(
                    size=(2, 2), interpolation="bilinear"
                )
                self.upsample_final = Conv2DTranspose(
                    filters=2,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="softmax",
                )

            def call(self, input, pool3, pool4):
                x = self.conv_7(input)
                x = self.drop(x)
                x = self.conv_8(x)

                x = self.upsample_pool4(x)
                x = Concatenate()([x, pool4])

                x = self.upsample_pool3(x)
                x = Concatenate()([x, pool3])

                x = UpSampling2D((2, 2))(x)
                x = UpSampling2D((2, 2))(x)

                x = self.upsample_final(x)
                return x

        class F1_Score(tf.keras.metrics.Metric):
            def __init__(self, name="f1_score", **kwargs):
                super().__init__(name=name, **kwargs)
                self.f1 = self.add_weight(name="f1", initializer="zeros")
                self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
                self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

            def update_state(self, y_true, y_pred, sample_weight=None):
                p = self.precision_fn(y_true, tf.argmax(y_pred, axis=-1))
                r = self.recall_fn(y_true, tf.argmax(y_pred, axis=-1))
                # since f1 is a variable, we use assign
                self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

            def result(self):
                return self.f1

            def reset_states(self):
                # we also need to reset the state of the precision and recall objects
                self.precision_fn.reset_states()
                self.recall_fn.reset_states()
                self.f1.assign(0)

        input_shape = (1024, 1024, 3)
        input_ = Input(shape=input_shape)

        vgg19_ = VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=input_,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )

        for layer in vgg19_.layers[:15]:
            layer.trainable = True

        pool3 = vgg19_.get_layer("block3_pool").output
        pool4 = vgg19_.get_layer("block4_pool").output

        conv_1_1_1 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            activation="relu",
            name="block6_conv1",
            kernel_regularizer=tf.keras.regularizers.l2(0.004),
        )(vgg19_.output)
        conv_1_1_1_drop = Dropout(0.8)(conv_1_1_1)

        conv_1_1_2 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            activation="relu",
            name="block6_conv2",
            kernel_regularizer=tf.keras.regularizers.l2(0.004),
        )(conv_1_1_1_drop)
        conv_1_1_2_drop = Dropout(0.8)(conv_1_1_2)

        table_mask_output = table_mask()(conv_1_1_2_drop, pool3, pool4)
        col_mask_output = col_mask()(conv_1_1_2_drop, pool3, pool4)

        model = Model(input_, [table_mask_output, col_mask_output])

        losses = {
            "table_mask": "sparse_categorical_crossentropy",
            "col_mask": "sparse_categorical_crossentropy",
        }

        metrics = [F1_Score(), "Accuracy"]

        # global init_lr
        init_lr = 0.0001

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr, epsilon=1e-8),
            loss=losses,
            metrics=metrics,
        )

        self.detect_model = tf.keras.models.load_model(
            self.path,
            custom_objects={
                "table_mask": table_mask,
                "col_mask": col_mask,
                "F1_Score": F1_Score,
            },
        )

    ## cropping table images form the table mask
    ## converts single page image into table images. individual image for every table that is detected using mode_15.h5
    def detected_table_images(self, img_path):
        if type(img_path) == Image.Image:
            img = np.array(img_path)
        elif isinstance(img_path, str):
            if "http" in img_path:
                image = Image.open(requests.get(img_path, stream=True).raw)
                img = np.array(image)
            else:
                image = Image.open(img_path)
                img = np.array(image)
        elif type(img_path) == jplugin.JpegImageFile:
            # img = Image.open(img_path)
            img = np.array(img_path)
        elif type(img_path) == bytes:
            img = Image.open(io.BytesIO(img_path))
            img = np.array(img)
        else:
            try:
                img = cv2.imread(img_path)
            except:
                imag = Image.open(img_path)
                img = np.array(imag)

        img = cv2.resize(img, (1024, 1024))
        img = np.expand_dims(img, axis=0)

        # Predict table mask
        table_mask, _ = self.detect_model.predict(img)

        # Threshold mask to get binary image
        table_mask = np.argmax(table_mask.squeeze(), axis=-1)
        table_mask = np.uint8(table_mask > 0.5)
        kernel = np.ones((5, 5), np.uint8)
        table_mask = cv2.dilate(table_mask, kernel, iterations=1)

        # Find contours of connected components
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Create the new folder if it doesn't exist
        # if not os.path.exists(new_folder_path):
        #     os.makedirs(new_folder_path)
        #     print("Created a new folder")
        table_imgs = []   
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            table_img = img[0][y : y + h, x : x + w]
            table_imgs.append(table_img)
        return table_imgs


class TableImagePreprocessing:
    def __init__(self):
        model_path = r"model_weights_13.pt"
        # mod = Classificationmodel(3)
        self.model = model

    ### rescaling the image
    def rescaling(self, image):
        image = Image.fromarray(image)
        resized_image = image
        w = image.size[0]
        h = image.size[1]
        if min(h, w) < 600:
            if w > h or w == h:
                factor = 600 / h
                h = 600
                w = int(factor * w)
            else:
                factor = 600 / w
                w = 600
                h = int(factor * h)

            resized_image = image.resize((w, h))
        return np.asarray(resized_image)

    ### removing the background color from the image
    def background_removal(self, image_path):
        image = image_path.copy()
        hsv_img = cv2.cvtColor(
            image_path, cv2.COLOR_BGR2HSV
        )  # convert image to HSV color space
        saturation_scale = 15  # increase saturation
        hsv_img[..., 1] = cv2.convertScaleAbs(
            hsv_img[..., 1], alpha=saturation_scale, beta=0
        )
        adjusted_saturation = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        hsv = cv2.cvtColor(adjusted_saturation, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 150, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        back_ground = []
        if len(contours) > 0:
            back_ground.append(1)
        else:
            back_ground.append(2)

        if 1 in back_ground:
            img = image_path.copy()
            hsv_img = cv2.cvtColor(
                img, cv2.COLOR_BGR2HSV
            )  # convert image to HSV color space
            saturation_scale = 0  # increase saturation
            hsv_img[..., 1] = cv2.convertScaleAbs(
                hsv_img[..., 1], alpha=saturation_scale, beta=0
            )
            adjusted_saturation = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            alpha = 1.4  # increase contrast
            beta = 0.5  # no brightness adjustment
            adjusted_contrast = cv2.convertScaleAbs(
                adjusted_saturation, alpha=alpha, beta=beta
            )
            return adjusted_contrast
        else:
            return image

    ### removing the horizontal and vertical lines form the image for partially bordered table
    def line_removal(self, image, original_image):
        pil_og_image = Image.fromarray(original_image)
        w, h = pil_og_image.size[0], pil_og_image.size[1]
        pil_image = Image.fromarray(image)
        w1, h1 = pil_image.size[0], pil_image.size[1]

        img = image.copy()
        img = cv2.resize(img, (w, h))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1, 1), 1)
        _, thresh = cv2.threshold(
            blur, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((5, 5), np.uint8)
        thresh1 = cv2.dilate(thresh, kernel=kernel, iterations=1)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 70))
        opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1, iterations=1)

        result = cv2.bitwise_and(thresh, cv2.bitwise_not(opening1))
        result1 = cv2.cvtColor(cv2.bitwise_not(result), cv2.COLOR_GRAY2RGB)

        gray = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        low_threshold = 50
        high_threshold = 100
        edges = cv2.Canny(blur, low_threshold, high_threshold)

        lines = cv2.HoughLines(edges, 1, np.pi / 20, 350)  # 20, 350
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(result1, (x1, y1), (x2, y2), (255, 255, 255), 4)

        img = cv2.resize(result1, (w1, h1))
        return img

    def sort_contours(self, cnts, method="left-to-right"):
        """Return sorted countours."""
        reverse = False
        k = 0
        if method in ["right-to-left", "bottom-to-top"]:
            reverse = True
        if method in ["top-to-bottom", "bottom-to-top"]:
            k = 1
        b_boxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, b_boxes) = zip(
            *sorted(zip(cnts, b_boxes), key=lambda b: b[1][k], reverse=reverse)
        )
        return (cnts, b_boxes)

    ## drawing  the horizontal and vertical lines
    def draw_lines(self, tbl_image):
        # tbl_image = img
        tbl_gray = cv2.cvtColor(tbl_image, cv2.COLOR_BGR2GRAY)
        tbl_thresh_bin = cv2.threshold(tbl_gray, 225, 255, cv2.THRESH_BINARY)[1]

        R = 3.0
        tbl_resized = resize(tbl_thresh_bin, width=int(tbl_image.shape[1] // R))

        def get_dividers(img, axis):
            """Return array indicies of white horizontal or vertical lines."""
            blank_lines = np.where(np.all(img == 255, axis=axis))[0]
            filtered_idx = np.where(np.diff(blank_lines) != 1)[0]
            if axis == 0:
                c = 0
                filtered_idx_1 = []
                for i in range(len(filtered_idx)):
                    if i == 0:
                        filtered_idx_1.append(filtered_idx[i])
                        continue
                    if filtered_idx[i] - filtered_idx[c] > 5:
                        filtered_idx_1.append(filtered_idx[i])
                    c = i
                filtered_idx = filtered_idx_1
            return blank_lines[filtered_idx]

        dims = tbl_image.shape[0], tbl_image.shape[1]
        tbl_str = np.zeros(dims, np.uint8)
        tbl_str = cv2.rectangle(tbl_str, (0, 0), (dims[1] - 1, dims[0] - 1), 255, 1)

        for a in [0, 1]:
            dividers = get_dividers(tbl_resized, a)
            start_point = [0, 0]
            if a == 0:
                end_point = [dims[0] - 5 , dims[0] -5 ]
            else:
                end_point = [dims[1] - 5 , dims[1] - 5 ]
            for i in dividers:
                i *= R
                start_point[a] = int(i)
                end_point[a] = int(i)
                cv2.line(tbl_str, tuple(start_point), tuple(end_point), 255, 1)

        contours, _ = cv2.findContours(tbl_str, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours, boundingBoxes = self.sort_contours(contours, method="top-to-bottom")

        # remove countours of the whole table
        bb_filtered = [
            list(t) for t in boundingBoxes if t[2] < dims[1] and t[3] < dims[0]
        ]

        # allocate countours in table-like structure
        rows = []
        columns = []

        for i, bb in enumerate(bb_filtered):
            if i == 0:
                columns.append(bb)
                previous = bb
            else:
                if bb[1] < previous[1] + previous[3] / 2:
                    columns.append(bb)
                    previous = bb
                    if i == len(bb_filtered) - 1:
                        rows.append(columns)
                else:
                    rows.append(columns)
                    columns = []
                    previous = bb
                    columns.append(bb)

        for row in rows:
            for column in row:
                x, y, w, h = column
                cv2.rectangle(tbl_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if len(row) > 0:
                x, y, w, h = row[0]
        return tbl_image

    # detection of horizontal and vertical lines to get the final boxes
    #finds the intersection points 
    def detect_horizontal_vertical_lines(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        last_folder_num = 0
        existing_folders = [folder for folder in os.listdir("./l_img") if folder.startswith("images")]
        if existing_folders:
            last_folder = max(existing_folders, key=lambda x: int(x.split("_")[1]))
            last_folder_num = int(last_folder.split("_")[1])
        # Increment the folder number
        new_folder_num = last_folder_num + 1
        new_folder_path = f"./l_img/images_{new_folder_num}"

        # Create the new folder if it doesn't exist
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print("created a new folder")

        cv2.imwrite(os.path.join(new_folder_path, "lines.png"),img)
        # image.save(os.path.join(new_folder_path, f"image{num}.png"))
        # thresholding the image to a binary image
        _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # inverting the image
        img_bin = 255 - img_bin
        # Length(width) of kernel as 100th of total width
        kernel_len = np.array(img).shape[1] // 100
        # Defining a vertical kernel to detect all vertical lines of image
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        # Defining a horizontal kernel to detect all horizontal lines of image
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        # A kernel of 2x2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
        image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

        img_vh = cv2.addWeighted(vertical_lines, 0.1, horizontal_lines, 0.1, 0.0)
        # Eroding and thesholding the image
        img_vh = cv2.erode(~img_vh, kernel, iterations=2)
        _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bitxor = cv2.bitwise_xor(img, img_vh)
        bitnot = cv2.bitwise_not(bitxor)

        contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort all the contours by top to bottom.
        contours, boundingBoxes = self.sort_contours(contours, method="top-to-bottom")

        heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
        # Get mean of heights
        mean = np.mean(heights)

        box = []
        # Get position (x,y), width and height for every contour and show the contour on image
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if img.shape[0] < 700 and img.shape[1] < 1000:
                if w < 1000 and h < 500:
                    image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    box.append([x, y, w, h])
            elif img.shape[0] < 700 and img.shape[1] > 1000:
                if w < 2100 and h < 500:
                    image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    box.append([x, y, w, h])
            elif img.shape[0] < 1000 and img.shape[1] > 700:
                if w < 500 and h < 2100:
                    image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    box.append([x, y, w, h])
            else:
                if w < 2100 and h < 1000:
                    image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    box.append([x, y, w, h])

        row = []
        column = []
        j = 0
        for i in range(len(box)):
            if i == 0:
                column.append(box[i])
                previous = box[i]
            else:
                if box[i][1] <= previous[1] + mean / 2:
                    column.append(box[i])
                    previous = box[i]
                    if i == len(box) - 1:
                        row.append(column)
                else:
                    row.append(column)
                    column = []
                    previous = box[i]
                    column.append(box[i])

        countcol = 0
        for i in range(len(row)):
            countcol = len(row[i])
            if countcol > countcol:
                countcol = countcol

            center = [
                int(row[i][j][0] + row[i][j][2] / 2)
                for j in range(len(row[i]))
                if row[0]
            ]
            center = np.array(center)
            center.sort()

        finalboxes = []
        for i in range(len(row)):
            ls = []
            for k in range(countcol):
                ls.append([])
            for j in range(len(row[i])):
                diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                ls[indexing].append(row[i][j])
            finalboxes.append(ls)

        return self.ocr(finalboxes, bitnot, row, countcol)

    #### ocr from the finalboxes
    def ocr(self, finalboxes, bitnot, row, countcol):
        outer = []
        sum = 0 
        for i in range(len(finalboxes)):
            for j in range(len(finalboxes[i])):
                inner = ""
                if len(finalboxes[i][j]) == 0:
                    outer.append(" ")
                else:
                    for k in range(len(finalboxes[i][j])):
                        y, x, w, h = (
                            finalboxes[i][j][k][0],
                            finalboxes[i][j][k][1],
                            finalboxes[i][j][k][2],
                            finalboxes[i][j][k][3],
                        )
                        finalimg = bitnot[x : x + h, y : y + w]
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                        border = cv2.copyMakeBorder(
                            finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255]
                        )
                        resizing = cv2.resize(
                            border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                        )
                        # dilation = cv2.dilate(resizing, kernel, iterations=1)
                        erosion = cv2.erode(resizing, kernel, iterations=1)
                        
                        cv2.imwrite(f"./e_img/er_img{sum}.png", erosion)
                        sum += 1 
                        out = pytesseract.image_to_string(erosion, config="--psm 6")
                        if len(out) == 0:
                            out = pytesseract.image_to_string(erosion, config="--psm 3")
                        inner = inner + " " + out
                        inner = inner.encode("utf-8").decode("utf-8")
                        inner = inner.encode("utf-8").decode("unicode_escape")
                    outer.append(inner)

        arr = np.array(outer)
        dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
        replace_func = lambda x: x.replace("\n", " ").replace("\f", " ")
        dataframe = dataframe.applymap(replace_func)
        dataframe.replace(r"^\s*$", np.nan, regex=True, inplace=True)

        dataframe.dropna(axis=1, how="all", inplace=True)
        c = 0
        for i in dataframe.copy().columns.to_list():
            dataframe.rename({i: str(c)}, axis=1, inplace=True)
            c += 1

        dataframe.dropna(axis=0, how="all", inplace=True)
        # data_dict = dataframe.to_dict(orient="records")
        dataframe.replace(np.nan, "", regex=True, inplace=True)
        # Format dictionary as JSON string
        return dataframe  # json.dumps({"Table 1": data_dict}, indent=4)

    def is_clean_image(self, img, threshold_contours=100):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_OTSU)[1]
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > threshold_contours:
            return img

    ### prediction for the classification model
    def table_border_classification_and_identification(self, image_path):
        image = Image.fromarray(image_path.astype("uint8"), "RGB")
        
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        t_img = transform(image).unsqueeze(0)
        self.model.eval()
        out = self.model(t_img)
        # out = model(t_img)
        _, pred = out.max(1)
        # print("Predicted")
        class_dict = {
            0: "bordered_table",
            1: "borderless_table",
            3: "partially_bordered_table",
            2: "others"
        }
        result = class_dict[pred.item()]
        print("result",result)
        # image.save("result.png")
        
        img = self.rescaling(image_path)
        img = self.background_removal(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_pixels = cv2.countNonZero(gray)
        total_pixels = img.shape[0] * img.shape[1]
        white_percentage = (white_pixels / total_pixels) * 100
        # if not a Complete white image, means if has text
        print("white_percentage",white_percentage)
        # if not a Complete white image, means if has text
        if white_percentage < 99.99:
            if result == "partially_bordered_table":
                img = self.line_removal(img, image_path)
                img = self.is_clean_image(img)
                
            if img is not None and result in ["borderless_table", "partially_bordered_table"]:
                img = self.draw_lines(img)
                img = self.is_clean_image(img)
            
            if img is not None:
                return self.detect_horizontal_vertical_lines(img)

class TableExtraction(TableImagePreprocessing, TableDetectionInImage):
    def __init__(self, input_file, file_name):
        TableDetectionInImage.__init__(self)
        TableImagePreprocessing.__init__(self)

        self.input_file = input_file

        try:
            ff = os.path.splitext(file_name)
        except:
            ff = str(file_name).split("/")[-1]

        if ff[1] in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.file_type = "image"
        elif ff[1] == ".pdf":
            self.file_type = "pdf"
        else:
            self.file_type = "file type not recognized"

    def get_table_extractions(self, image):
        table_img_list = self.detected_table_images(image)
        print(len(table_img_list))
        # print(table_img_list)
        extractions = []
        import os
        last_folder_num = 0
        existing_folders = [folder for folder in os.listdir("./img") if folder.startswith("images")]
        if existing_folders:
            last_folder = max(existing_folders, key=lambda x: int(x.split("_")[1]))
            last_folder_num = int(last_folder.split("_")[1])
        # Increment the folder number
        new_folder_num = last_folder_num + 1
        new_folder_path = f"./img/images_{new_folder_num}"

        # Create the new folder if it doesn't exist
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print("created a new folder")
        for num, table_img in enumerate(table_img_list):  
            image = Image.fromarray(table_img)
            # Save the image with an iterative name
            image.save(os.path.join(new_folder_path, f"image{num}.png"))
            extraction = self.table_border_classification_and_identification(table_img)
            # if len(extraction) != 0:#
            if extraction is not None:
                extractions.append(
                    {
                        f"Table {num+1}": json.dumps(
                            extraction.to_dict(orient="records"), indent=4
                        )
                    }
                )

        def remove_newlines(data):
            for item in data:
                for key, value in item.items():
                    if isinstance(value, str):
                        item[key] = value.replace("\n", "")
                        item[key] = json.loads(value)
            return data

        extractions = remove_newlines(extractions)

        return extractions

    def extract_table(self):
        if self.file_type == "pdf":
            pdf_path = self.input_file

            # Convert PDF pages to PIL image objects
            images = convert_from_path(pdf_path, fmt="jpeg")
            extractions = []
            for i, img in enumerate(images):
                print("page no ",i)
                extract = self.get_table_extractions(img)
                extract = {"Page " + str(i + 1): extract}
                extractions.append(extract)
                break
            return extractions

        elif self.file_type == "image":
            extractions = self.get_table_extractions(self.input_file)
            extractions = [{"Page 1": extractions}]

            return extractions

        else:
            return "Given file is neither an Image nor a PDF!", None


import __main__

import os

# Specify the folder names
folder_names = ['img', 'e_img','l_img']

# Define the parent directory path where the folders will be created
parent_directory = './'

# Create the folders
for folder_name in folder_names:
    folder_path = os.path.join(parent_directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)
setattr(__main__, "Classificationmodel", Classificationmodel)
model = torch.load(r"model_weights_13.pt", map_location=torch.device("cpu"))
# obj = TableExtraction(r"C:\Users\Admin\Downloads\table_ocr_project\15032-5280-FullBook.pdf")
obj = TableExtraction(r"OCR_SANTOSH/291169012_suoypoa1rvi5puntbojalx12.pdf","291169012_suoypoa1rvi5puntbojalx12.pdf")
result = obj.extract_table()
print("RESULT")
print(result)