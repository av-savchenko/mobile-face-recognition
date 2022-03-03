import sys,os
import torch
from torchvision import transforms,models
from torch.utils.mobile_optimizer import optimize_for_mobile

import json
sys.path.append("../once-for-all")
from ofa.imagenet_classification.networks.mobilenet_v3 import MobileNetV3
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3

def get_model_artifacts(net_folder,model_name):
    config_filepath = os.path.join(net_folder, model_name,f"{model_name}.config")
    test_pt_filepath = os.path.join(net_folder, model_name,f"{model_name}.pth")
    model_config = json.load(open(config_filepath, 'r'))
    test_checkpoint = torch.load(test_pt_filepath, map_location=torch.device('cpu'))
    filter_end_fn = lambda x : not x.endswith('total_ops') and not x.endswith('total_params')
    filter_start_fn = lambda x : not x.startswith('total_ops') and not x.startswith('total_params')
    filtered_state_dict = {key:value for key,value in test_checkpoint['state_dict'].items() if filter_start_fn(key) and filter_end_fn(key)}
    model=MobileNetV3.build_from_config(model_config)
    model.load_state_dict(filtered_state_dict)
    #torch.save(model,model_name+'.pt')
    return model

if __name__ == '__main__':
    output_model_dir='../mobile/FaceMatcher/app/src/main/assets'
    INPUT_SIZE=224
    example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    model_dir='../models/ofa_subnets'
    for model_name in os.listdir(model_dir):
        filename=os.path.join(output_model_dir,model_name)
        if not os.path.exists(filename+'.ptl'):
            print(model_name)
            model=get_model_artifacts(model_dir,model_name)
            model.classifier.linear=torch.nn.Identity()
            model.eval()
            traced_script_module = torch.jit.trace(model, example)
            traced_script_module_optimized = optimize_for_mobile(traced_script_module)
            traced_script_module_optimized._save_for_lite_interpreter(filename+'.ptl')
