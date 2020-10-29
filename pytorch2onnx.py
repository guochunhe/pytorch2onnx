import torch
# import trochvision
import torch.utils.data
import argparse
import onnxruntime
from mtcnn.core.models import PNet,RNet,ONet # 加载自己的网络模型
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import numpy as np
from torch.autograd import Variable
from onnxruntime.datasets import get_example


def main(args):
    # p_model_path = '/data/guoch_workspace/mtcnn-pytorch-master/model_store/pnet_epoch.pt'
    # r_model_path = '/data/guoch_workspace/mtcnn-pytorch-master/model_store/rnet_epoch.pt'
    # o_model_path = '/data/guoch_workspace/mtcnn-pytorch-master/model_store/onet_epoch.pt'
    # print("the version of torch is {}".format(torch.__version__))
    dummy_input=getInput(args.img_size)#获得网络的输入
    # 加载模型
    model = PNet()
    #model = RNet()
    #model = ONet()
    model.load_state_dict(torch.load(args.model_path))
    #model_dict =  model.state_dict()
    #model_dict = pnet.load_state_dict(torch.load(p_model_path))
    # if args.model_path:
    #     if os.path.isfile(args.model_path):
    #         print(("=> start loading checkpoint '{}'".format(args.model_path)))
    #         # state_dict = torch.load(args.model_path)
    #         # print("the best acc is {} in epoch:{}".format(
    #         #     state_dict['epoch_acc'], state_dict['epoch']))
    #         # params = state_dict["model_state_dict"]
    #         # # params={k:v for k,v in state_dict.items() if k in  model_dict.keys()}
    #         # # model_dict.update(params)
    #         # # model.load_state_dict(model_dict)
    #         model.load_state_dict(args.model_path)
    #         print("load cls model successfully")
    #     else:
    #         print(("=> no checkpoint found at '{}'".format(args.model_path)))
    #         return
    model.to('cpu')
    model.eval()
    pre=model(dummy_input)
    print("the pre:{}".format(pre))
    #保存onnx模型
    torch2onnx(args,model,dummy_input)

def getInput(img_size):
    input = cv2.imread('/data/guoch_workspace/mtcnn-pytorch-master/test_dl_data/sample.jpg')
    input = cv2.resize(input, (12, 12))  # hwc bgr  # pnet12, 12 / rnet 24 24 / onet 48 48
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)  # hwc rgb
    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input = np.transpose(input, (2, 0, 1)).astype(np.float32)  # chw rgb
    # input=input/255.0
    print("befor the input[0,0,0]:{}".format(input[0, 0, 0]))
    print("the size of input[0,...] is {}".format(input[0, ...].shape))
    print("the size of input[1,...] is {}".format(input[1, ...].shape))
    print("the size of input[2,...] is {}".format(input[2, ...].shape))
    input[0, ...] = ((input[0, ...]/255.0)-0.485)/0.229
    input[1, ...] = ((input[1, ...]/255.0)-0.456)/0.224
    input[2, ...] = ((input[2, ...]/255.0)-0.406)/0.225
    print("after input[0,0,0]:{}".format(input[0, 0, 0]))

    now_image1 = Variable(torch.from_numpy(input))
    dummy_input = now_image1.unsqueeze(0)
    return dummy_input


def torch2onnx(args,model,dummy_input):
    input_names = ['input']#模型输入的name
    output_names = ['output']#模型输出的name
    # return
    torch_out = torch.onnx._export(model, dummy_input, os.path.join(args.save_model_path,"pnet.onnx"),
                                   verbose=True, input_names=input_names, output_names=output_names)
    # test onnx model
    example_model = get_example(os.path.join(args.save_model_path,"pnet.onnx"))
    session = onnxruntime.InferenceSession(example_model)
    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name
    print('Input Name:', input_name)
    result = session.run([], {input_name: dummy_input.data.numpy()})
    # np.testing.assert_almost_equal(
    #     torch_out.data.cpu().numpy(), result[0], decimal=3)
    print("the result is {}".format(result))
    #结果对比--有点精度上的损失
    # pytorch tensor([[ 5.8738, -5.4470]], device='cuda:0')
    # onnx [ 5.6525207 -5.2962923]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch model to onnx and ncnn")
    parser.add_argument('--model_path', type=str, default="/data/guoch_workspace/mtcnn/mtcnn-pytorch-zh/model_store/v2/bs512_lre2/pnet_epoch_9.pt",
                        help="For training from one model_file")
    parser.add_argument('--save_model_path', type=str, default="/data/guoch_workspace/mtcnn/mtcnn-pytorch-zh/model_store/v2/rnet/bs128_lre3/",
                        help="For training from one model_file")
    # parser.add_argument('--onnx_model_path', type=str, default="/data/guoch_workspace/mtcnn-pytorch-master/model_store/copy/rnet_epoch.onnx",
    #                     help="For training from one model_file")
    parser.add_argument('--img_size', type=int, default=48,
                        help="the image size of model input")
    args = parser.parse_args()
    main(args)

