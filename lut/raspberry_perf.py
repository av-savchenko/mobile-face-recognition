import os
import numpy as np
import time

import tflite_runtime.interpreter as tflite

NUM_ATTEMPTS=1000

def measure_tflite_perf(model_path):
    interpreter = tflite.Interpreter(model_path)#,num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32
    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    x=np.float32(2*np.random.rand(*input_details[0]['shape'])-1)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    #interpreter.set_tensor(input_details[0]['index'], x)

    total_times=[]
    for _ in range(NUM_ATTEMPTS):
        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()
        total_time=(stop_time - start_time)
        total_times.append(1000*total_time)

    return np.mean(total_times),np.std(total_times)

def measure_assets_perf():
    root_dir='../mobile/FaceMatcher/app/src/main/assets'
    for f in os.listdir(root_dir):
        if f.endswith('.tflite'):
            mean,std=measure_tflite_perf(os.path.join(root_dir,f))
            print(f,mean,std)

def compute_lut():
    archInfoPath = 'references/tflite_lut_arch2info.txt'
    outLutFileName = "raspberry_lookup_table.txt";

    with open(archInfoPath) as file:
        with open(outLutFileName, 'a') as out:
            for line in file:
                #print()
                #model_path='expanded_conv-input_colon_7x7x192-output_colon_7x7x192-expand_colon_768-kernel_colon_7-stride_colon_1-idskip_colon_1-se_colon_1-hs_colon_1.tflite'
                vals=line.rstrip().split(' ')
                model_path=vals[1].replace(':','_colon_')
                print(model_path)
                mean,std=measure_tflite_perf(os.path.join('mobilenetv3_tflite_model_parts',model_path))
                #print(mean,std)
                output_line = vals[0] + " " + str(NUM_ATTEMPTS) + " " + str(mean) + " " + str(std) + "\n";
                out.write(output_line)
                #output_data = interpreter.get_tensor(output_details[0]['index'])

#compute_lut()
measure_assets_perf()
