# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
import tvm.testing
from tvm import relax as rx
from tvm.script import relax as R
from tvm.script import tir as T
from pathlib import Path
import sys
import json

# Hackery to enable importing of utils from ci/scripts/jenkins
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT / "tests" / "python" / "frontend" / "onnx"))

from test_forward import verify_with_ort
# import onnx

from onnx import hub


BASE = "https://github.com/onnx/models/raw/main"
ONNX_MODELS = [
    # "vision/body_analysis/ultraface/models/version-RFB-320.tar.gz",
    # "vision/body_analysis/ultraface/models/version-RFB-640.tar.gz",
    # "vision/body_analysis/arcface/model/arcfaceresnet100-8.tar.gz",
    # "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.tar.gz",
    # "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.tar.gz",
    # "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-2.tar.gz",
    # "vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/udnie-9.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/mosaic-9.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/candy-8.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/rain-princess-8.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/candy-9.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/udnie-8.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/pointilism-8.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/mosaic-8.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/rain-princess-9.tar.gz",
    # "vision/style_transfer/fast_neural_style/model/pointilism-9.tar.gz",
    # "vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.tar.gz",
    # "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.tar.gz",
    # "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz",
    # "vision/object_detection_segmentation/fcn/model/fcn-resnet50-12-int8.tar.gz",
    # "vision/object_detection_segmentation/fcn/model/fcn-resnet101-11.tar.gz",
    # "vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.tar.gz",
    # "vision/object_detection_segmentation/fcn/model/fcn-resnet50-11.tar.gz",
    # "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12-int8.tar.gz",
    # "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.tar.gz",
    # "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.tar.gz",
    # "vision/object_detection_segmentation/ssd/model/ssd-10.tar.gz",
    # "vision/object_detection_segmentation/ssd/model/ssd-12-int8.tar.gz",
    # "vision/object_detection_segmentation/ssd/model/ssd-12.tar.gz",
    # "vision/object_detection_segmentation/yolov3/model/yolov3-10.tar.gz",
    # "vision/object_detection_segmentation/yolov3/model/yolov3-12-int8.tar.gz",
    # "vision/object_detection_segmentation/yolov3/model/yolov3-12.tar.gz",
    # "vision/object_detection_segmentation/retinanet/model/retinanet-9.tar.gz",
    # "vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.tar.gz",
    # "vision/object_detection_segmentation/yolov4/model/yolov4.tar.gz",
    # "vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12-int8.tar.gz",
    # "vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.tar.gz",
    # "vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.tar.gz",
    # "vision/object_detection_segmentation/duc/model/ResNet101-DUC-12-int8.tar.gz",
    # "vision/object_detection_segmentation/duc/model/ResNet101-DUC-12.tar.gz",
    # "vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.tar.gz",
    # "vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12-int8.tar.gz",
    # "vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.tar.gz",
    # "vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.tar.gz",
    # "vision/classification/efficientnet-lite4/model/efficientnet-lite4-11-int8.tar.gz",
    # "vision/classification/efficientnet-lite4/model/efficientnet-lite4-11-qdq.tar.gz",
    # "vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.tar.gz",
    # "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-7.tar.gz",
    # "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-6.tar.gz",
    # "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-8.tar.gz",
    # "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.tar.gz",
    # "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-3.tar.gz",
    # "vision/classification/densenet-121/model/densenet-12.tar.gz",
    # "vision/classification/densenet-121/model/densenet-6.tar.gz",
    # "vision/classification/densenet-121/model/densenet-3.tar.gz",
    # "vision/classification/densenet-121/model/densenet-9.tar.gz",
    # "vision/classification/densenet-121/model/densenet-8.tar.gz",
    # "vision/classification/densenet-121/model/densenet-12-int8.tar.gz",
    # "vision/classification/densenet-121/model/densenet-7.tar.gz",
    # "vision/classification/squeezenet/model/squeezenet1.0-13-qdq.tar.gz",
    # "vision/classification/squeezenet/model/squeezenet1.0-9.tar.gz",
    # "vision/classification/squeezenet/model/squeezenet1.0-7.tar.gz",
    # "vision/classification/squeezenet/model/squeezenet1.0-8.tar.gz",
    # "vision/classification/squeezenet/model/squeezenet1.0-3.tar.gz",
    # "vision/classification/squeezenet/model/squeezenet1.0-12.tar.gz",
    # "vision/classification/squeezenet/model/squeezenet1.1-7.tar.gz",
    # "vision/classification/squeezenet/model/squeezenet1.0-6.tar.gz",
    # "vision/classification/squeezenet/model/squeezenet1.0-12-int8.tar.gz",
    # "vision/classification/caffenet/model/caffenet-3.tar.gz",
    # "vision/classification/caffenet/model/caffenet-6.tar.gz",
    # "vision/classification/caffenet/model/caffenet-12.tar.gz",
    # "vision/classification/caffenet/model/caffenet-7.tar.gz",
    # "vision/classification/caffenet/model/caffenet-9.tar.gz",
    # "vision/classification/caffenet/model/caffenet-8.tar.gz",
    # "vision/classification/caffenet/model/caffenet-12-qdq.tar.gz",
    # "vision/classification/caffenet/model/caffenet-12-int8.tar.gz",
    # "vision/classification/mobilenet/model/mobilenetv2-12-int8.tar.gz",
    # "vision/classification/mobilenet/model/mobilenetv2-10.tar.gz",
    # "vision/classification/mobilenet/model/mobilenetv2-12.tar.gz",
    # "vision/classification/mobilenet/model/mobilenetv2-12-qdq.tar.gz",
    # "vision/classification/mobilenet/model/mobilenetv2-7.tar.gz",
    # "vision/classification/shufflenet/model/shufflenet-v2-12.tar.gz",
    # "vision/classification/shufflenet/model/shufflenet-8.tar.gz",
    # "vision/classification/shufflenet/model/shufflenet-v2-12-qdq.tar.gz",
    # "vision/classification/shufflenet/model/shufflenet-v2-12-int8.tar.gz",
    # "vision/classification/shufflenet/model/shufflenet-6.tar.gz",
    # "vision/classification/shufflenet/model/shufflenet-v2-10.tar.gz",
    # "vision/classification/shufflenet/model/shufflenet-9.tar.gz",
    # "vision/classification/shufflenet/model/shufflenet-7.tar.gz",
    # "vision/classification/shufflenet/model/shufflenet-3.tar.gz",
    # "vision/classification/resnet/model/resnet152-v1-7.tar.gz",
    # "vision/classification/resnet/model/resnet34-v1-7.tar.gz",
    # "vision/classification/resnet/model/resnet50-v1-12.tar.gz",
    # "vision/classification/resnet/model/resnet152-v2-7.tar.gz",
    # "vision/classification/resnet/model/resnet50-v1-7.tar.gz",
    # "vision/classification/resnet/model/resnet50-v1-12-int8.tar.gz",
    # "vision/classification/resnet/model/resnet50-caffe2-v1-8.tar.gz",
    # "vision/classification/resnet/model/resnet18-v2-7.tar.gz",
    # "vision/classification/resnet/model/resnet50-v2-7.tar.gz",
    # "vision/classification/resnet/model/resnet50-caffe2-v1-7.tar.gz",
    # "vision/classification/resnet/model/resnet50-v1-12-qdq.tar.gz",
    # "vision/classification/resnet/model/resnet50-caffe2-v1-3.tar.gz",
    # "vision/classification/resnet/model/resnet101-v1-7.tar.gz",
    # "vision/classification/resnet/model/resnet34-v2-7.tar.gz",
    # "vision/classification/resnet/model/resnet18-v1-7.tar.gz",
    # "vision/classification/resnet/model/resnet50-caffe2-v1-6.tar.gz",
    # "vision/classification/resnet/model/resnet50-caffe2-v1-9.tar.gz",
    # "vision/classification/resnet/model/resnet101-v2-7.tar.gz",
    "vision/classification/alexnet/model/bvlcalexnet-12.tar.gz",
    # "vision/classification/alexnet/model/bvlcalexnet-9.tar.gz",
    # "vision/classification/alexnet/model/bvlcalexnet-12-qdq.tar.gz",
    # "vision/classification/alexnet/model/bvlcalexnet-12-int8.tar.gz",
    # "vision/classification/alexnet/model/bvlcalexnet-8.tar.gz",
    # "vision/classification/alexnet/model/bvlcalexnet-3.tar.gz",
    # "vision/classification/alexnet/model/bvlcalexnet-6.tar.gz",
    # "vision/classification/alexnet/model/bvlcalexnet-7.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-7.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-3.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-6.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-8.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.tar.gz",
    # "vision/classification/inception_and_googlenet/googlenet/model/googlenet-7.tar.gz",
    # "vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.tar.gz",
    # "vision/classification/inception_and_googlenet/googlenet/model/googlenet-6.tar.gz",
    # "vision/classification/inception_and_googlenet/googlenet/model/googlenet-3.tar.gz",
    # "vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.tar.gz",
    # "vision/classification/inception_and_googlenet/googlenet/model/googlenet-12-qdq.tar.gz",
    # "vision/classification/inception_and_googlenet/googlenet/model/googlenet-8.tar.gz",
    # "vision/classification/inception_and_googlenet/googlenet/model/googlenet-12-int8.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-12.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-12-int8.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-7.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-6.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-3.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-8.tar.gz",
    # "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-12-qdq.tar.gz",
    # "vision/classification/vgg/model/vgg16-7.tar.gz",
    # "vision/classification/vgg/model/vgg19-bn-7.tar.gz",
    # "vision/classification/vgg/model/vgg16-bn-7.tar.gz",
    # "vision/classification/vgg/model/vgg19-caffe2-3.tar.gz",
    # "vision/classification/vgg/model/vgg16-12.tar.gz",
    # "vision/classification/vgg/model/vgg19-caffe2-8.tar.gz",
    # "vision/classification/vgg/model/vgg19-caffe2-6.tar.gz",
    # "vision/classification/vgg/model/vgg19-caffe2-9.tar.gz",
    # "vision/classification/vgg/model/vgg16-12-int8.tar.gz",
    # "vision/classification/vgg/model/vgg19-caffe2-7.tar.gz",
    # "vision/classification/vgg/model/vgg19-7.tar.gz",
    # "vision/classification/mnist/model/mnist-12-int8.tar.gz",
    # "vision/classification/mnist/model/mnist-12.tar.gz",
    # "vision/classification/mnist/model/mnist-8.tar.gz",
    # "vision/classification/mnist/model/mnist-7.tar.gz",
    # "vision/classification/mnist/model/mnist-1.tar.gz",
    # "vision/classification/zfnet-512/model/zfnet512-3.tar.gz",
    # "vision/classification/zfnet-512/model/zfnet512-7.tar.gz",
    # "vision/classification/zfnet-512/model/zfnet512-12-int8.tar.gz",
    # "vision/classification/zfnet-512/model/zfnet512-8.tar.gz",
    # "vision/classification/zfnet-512/model/zfnet512-9.tar.gz",
    # "vision/classification/zfnet-512/model/zfnet512-6.tar.gz",
    # "vision/classification/zfnet-512/model/zfnet512-12.tar.gz",
    # "text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.tar.gz",
    # "text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz",
    # "text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.tar.gz",
    # "text/machine_comprehension/t5/model/t5-encoder-12.tar.gz",
    # "text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.tar.gz",
    # "text/machine_comprehension/roberta/model/roberta-sequence-classification-9.tar.gz",
    # "text/machine_comprehension/roberta/model/roberta-base-11.tar.gz",
    # "text/machine_comprehension/bert-squad/model/bertsquad-10.tar.gz",
    # "text/machine_comprehension/bert-squad/model/bertsquad-12-int8.tar.gz",
    # "text/machine_comprehension/bert-squad/model/bertsquad-12.tar.gz",
    # "text/machine_comprehension/bert-squad/model/bertsquad-8.tar.gz",
]

_manifest = None

def manifest():
    global _manifest
    if _manifest is None:
        with open(Path(__file__).resolve().parent / "ONNX_HUB_MANIFEST.json") as f:
            _manifest = json.load(f)

    return _manifest

def find_model(name):
    for model in manifest():
        if model["model_path"] == name:
            return model
    raise ValueError(f"{name} not found")


def test_coverage():
    metadata = find_model("vision/classification/alexnet/model/bvlcalexnet-12.onnx")
    input_shapes = [metadata["metadata"]["io_ports"]["inputs"][0]["shape"]]
    model = hub.load("alexnet", repo="onnx/models:8e893eb39b131f6d3970be6ebd525327d3df34ea", silent=True)
    verify_with_ort(model, input_shapes=input_shapes)



if __name__ == "__main__":
    tvm.testing.main()
    
