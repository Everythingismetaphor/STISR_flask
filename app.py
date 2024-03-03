import os
import sys
import json
import cv2
import time
import torch
from PIL import Image
import io
from models.crnn import crnn
from torchvision import transforms
import  torchvision
import matplotlib.pyplot as plt
from models.tbsrn_dl_esa import TBSRN
from models.dual_vit import TSRN_TL_TRANS
#from models.model_v1_CBAM import efficientnet_b1_2 as create_model_cbam
# Flask

from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_cors import CORS
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
# from models.eval import cutimage



# Declare a flask app
app = Flask(__name__)
CORS(app)  # 解决跨域问题

# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# # model = MobileNetV2(weights='imagenet')
UPLOAD_FOLDER = 'static/input/'

OUTPUT_FOLDER = 'static/output/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print('Model loaded. Check http://127.0.0.1:5001/')
print('Model loaded. Check http://127.0.0.1/ or http://bi.zjnu.edu.cn/')

###文本识别器初始化
def CRNN_init( recognizer_path=None, opt=None):
    model = crnn.CRNN(32, 1, 37, 256)
    model = model.to(device)
    #使用微调后的识别器性能更好
    model_path = "models/recognizer_best_0.pth"
    print('loading pretrained crnn model from %s' % model_path)
    stat_dict = torch.load(model_path)
    # print("stat_dict:", stat_dict.keys())
    # if recognizer_path is None:
    try:
        model.load_state_dict(stat_dict)
    # else:
    except Exception:
        model = stat_dict
    # model #.eval()
    # model.eval()

    return model

####crnn输入图像预处理
def parse_crnn_data(imgs_input):
    in_width =  100
    batch_size = imgs_input.shape[0]
    imgs_input = torch.nn.functional.interpolate(imgs_input[:, :3, ...], (32, in_width), mode='bicubic')

    # imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
    R = imgs_input[:, 0:1, :, :]
    G = imgs_input[:, 1:2, :, :]
    B = imgs_input[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor
def image_to_base64(img_array):
    # 在实际应用中，你需要使用适当的库将图像数组转换为 base64 编码的字符串
    # 这里使用示例库 base64 将 numpy 数组转换为 base64 编码的字符串
    import base64
    from io import BytesIO
    from PIL import Image

    #img = Image.fromarray(img_array.astype('uint8'))  # 假设 img_array 是 uint8 类型的图像数组
    img_array = img_array.to('cpu').detach().numpy()
    img = Image.fromarray(img_array.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")  # 保存图像为 PNG 格式，你可以根据需要修改格式
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_base64
def image_sr_mothod2(img):
    def transform_(img):
        img = img.resize((64, 16), Image.BICUBIC)
        img_tensor = transforms.ToTensor()(img)
        mask = img.convert('L')
        thres = np.array(mask).mean()
        mask = mask.point(lambda x: 0 if x > thres else 255)
        mask = transforms.ToTensor()(mask)
        img_tensor = torch.cat((img_tensor, mask), 0)
        img_tensor = img_tensor.unsqueeze(0)#增加batch维度
        return img_tensor
    # [N, C, H, W] 图片处理
    images_lr = transform_(img)
    # create model
    model = TSRN_TL_TRANS(scale_factor=2, width=128, height=32,
                               STN=True, mask=True, srb_nums=5, hidden_units=32)

    #model to gpu
    model = model.to(device)
    crnn = CRNN_init()
    crnn.eval()
    aster_dict_lr = parse_crnn_data(images_lr[:, :3, :, :]).to(device)
    label_vecs_logits = crnn(aster_dict_lr)
    label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
    label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
    drop_vec = torch.ones(images_lr.shape[0]).float()
    drop_vec[:int(images_lr.shape[0] // 4)] = 0.
    drop_vec = drop_vec.to(device)
    label_vecs_final = label_vecs_final * drop_vec.view(-1, 1, 1, 1)
    # load model weights
    # model = create_model(num_classes=10).to(device)
    model_weight_path = "models/model_method2.pth"
    #model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.load_state_dict( {k.replace("module.", ""): v for k, v in torch.load(model_weight_path)['state_dict_G'].items()}, strict=False)
    model.eval()

    with torch.no_grad():
        # predict class
        sr_beigin = time.time()
        #output = torch.squeeze(model(images_lr.to(device))).cpu()
        images_sr = model(images_lr.to(device), label_vecs_final.to(device))
        sr_end  =   time.time()
        sr_time = sr_end - sr_beigin
    print("方法2图片超分完成!耗时: {:.2f} 秒".format(sr_time))
    return images_sr


def image_sr_mothod1(img):

    def transform_(img):
        img = img.resize((64, 16), Image.BICUBIC)
        img_tensor = transforms.ToTensor()(img)
        mask = img.convert('L')
        thres = np.array(mask).mean()
        mask = mask.point(lambda x: 0 if x > thres else 255)
        mask = transforms.ToTensor()(mask)
        img_tensor = torch.cat((img_tensor, mask), 0)
        img_tensor = img_tensor.unsqueeze(0)#增加batch维度
        return img_tensor
    # create model
    model = TBSRN(scale_factor=2, width=128, height=32,
                               STN=True, mask=True, srb_nums=5, hidden_units=32)
    #model to gpu
    model = model.to(device)
    # load model weights
    # model = create_model(num_classes=10).to(device)
    model_weight_path = "models/aster_model_best.pth"
    #model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.load_state_dict( {k.replace("module.", ""): v for k, v in torch.load(model_weight_path)['state_dict_G'].items()}, strict=False)
    model.eval()
    # [N, C, H, W] 图片处理
    images_lr = transform_(img)
    with torch.no_grad():
        # predict class
        sr_beigin = time.time()
        #output = torch.squeeze(model(images_lr.to(device))).cpu()
        images_sr = model(images_lr.to(device))
        sr_end  =   time.time()
        sr_time = sr_end - sr_beigin
    print("方法1图片超分完成!耗时: {:.2f} 秒".format(sr_time))
    return images_sr



@app.route('/', methods=['GET'])  # 装饰器 界面1
def index():
    # Main page
    return render_template('index.html')


@app.route('/sr1', methods=['GET', 'POST'])  # 装饰器 界面predict
def predict_easy():
    if request.method == 'POST':
        # Get the image from post request
        img_bytes = request.files["file"]
        # 字节对象转化
        img_bytes = img_bytes.read()
        img = Image.open(io.BytesIO(img_bytes))
        print("方法1开始处理！")
        img_sr = image_sr_mothod1(img).squeeze(0)
        img_sr = img_sr[:3, :, :]
        # #保存sr图片
        torchvision.utils.save_image(img_sr, OUTPUT_FOLDER+"sr_image.png")
        # 将 img_sr 转换为 base64 编码的图像数据
        #img_base64 = image_to_base64(img_sr)
        # 返回 base64 编码的图像数据
        #return jsonify({'img_base64': img_base64})
        # 处理POST请求中的图像，假设图像路径为html/static/output/sr_image.png
        image_path = 'static/output/sr_image.png'
        return send_file(image_path, mimetype='image/png')

    return" ERROR"
@app.route('/sr2', methods=['GET', 'POST'])  # 装饰器 界面predict
def predict_hard():
    if request.method == 'POST':
        # Get the image from post request
        img_bytes = request.files["file"]
        # 字节对象转化
        img_bytes = img_bytes.read()
        img = Image.open(io.BytesIO(img_bytes))
        print("方法2开始处理！")
        img_sr = image_sr_mothod2(img).squeeze(0)
        img_sr = img_sr[:3, :, :]
        # #保存sr图片
        torchvision.utils.save_image(img_sr, OUTPUT_FOLDER+"sr_image.png")
        # 将 img_sr 转换为 base64 编码的图像数据
        #img_base64 = image_to_base64(img_sr)
        # 返回 base64 编码的图像数据
        #return jsonify({'img_base64': img_base64})
        # 处理POST请求中的图像，假设图像路径为html/static/output/sr_image.png
        image_path = 'static/output/sr_image.png'
        return send_file(image_path, mimetype='image/png')

    return" ERROR"


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 80), app)
    http_server.serve_forever()



    # ####测试
    # img = Image.open(f"test_img/cropped_images_lr/lrw_srr_lr_sr_hr_452_dacowy_discovery_discovery_discovery_.png")
    # img_sr = image_sr_mothod2(img).squeeze(0)
    # #保存sr图片
    # torchvision.utils.save_image(img_sr[:3, :, :],
    #                              "sr_image.png")