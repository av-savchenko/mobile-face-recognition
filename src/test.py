#timm 0.4.5
import argparse
import sys
import os.path
import os
import math
import datetime, time
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
#from scipy.misc import imread
from matplotlib.pyplot import imread
import cv2
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from PIL import Image

import torch

np.random.seed(123)  # for reproducibility

DATASET_PATH='D:/datasets/lfw_ytf/lfw_mtcnn_aligned'#_faces'#_cropped #mtcnn #mtcnn_aligned
#DATASET_PATH='D:/datasets/lfw_ytf/lfw_mtcnn'
#DATASET_PATH='D:/datasets/lfw_ytf/lfw_cropped'
#DATASET_PATH='D:/datasets/lfw_ytf/lfw_faces'


TF, INSIGHTFACE, TORCH=0, 1, 2
use_framework=TORCH #TF #INSIGHTFACE #
MY_TORCH_MODEL=True
OFA_MODEL=True

if use_framework==TORCH and OFA_MODEL:
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



if use_framework==TF:
    import tensorflow as tf

img_extensions=['.jpg','.jpeg','.png']
def is_image(path):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in img_extensions

def get_files(db_dir):
    return [[d,os.path.join(d,f)] for d in next(os.walk(db_dir))[1] for f in next(os.walk(os.path.join(db_dir,d)))[2] if not f.startswith(".") and is_image(f)]

def load_graph(frozen_graph_filename, prefix=''):
    with tf.io.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph

class TensorFlowInference:
    def __init__(self,frozen_graph_filename,input_tensor,output_tensor,learning_phase_tensor=None, convert2BGR=True, imageNetUtilsMean=True,additional_input_value=0):
        graph = load_graph(frozen_graph_filename,'')
        print([n.name for n in graph.as_graph_def().node if 'input' in n.name])
        
        graph_op_list=list(graph.get_operations())
        print([n.name for n in graph_op_list if 'keras_learning' in n.name])
        
        self.tf_sess=tf.compat.v1.Session(graph=graph)
        
        self.tf_input_image = graph.get_tensor_by_name(input_tensor)
        print('tf_input_image=',self.tf_input_image)
        self.tf_output_features = graph.get_tensor_by_name(output_tensor)
        print('tf_output_features=',self.tf_output_features)
        self.tf_learning_phase = graph.get_tensor_by_name(learning_phase_tensor) if learning_phase_tensor else None;
        print('tf_learning_phase=',self.tf_learning_phase)
        if self.tf_input_image.shape.dims is None:
            w=h=160
        else:
            _,w,h,_=self.tf_input_image.shape
        self.w,self.h=int(w),int(h)
        print ('input w,h',self.w,self.h,' output shape:',self.tf_output_features.shape,'convert2BGR:',convert2BGR, 'imageNetUtilsMean:',imageNetUtilsMean)
        #for n in graph.as_graph_def().node:
        #    print(n.name, n.op)
        #sys.exit(0)

        self.convert2BGR=convert2BGR
        self.imageNetUtilsMean=imageNetUtilsMean
        self.additional_input_value=additional_input_value

    def preprocess_image(self,img_filepath,crop_center):
        if crop_center:
            orig_w,orig_h=250,250
            img = imread(img_filepath)#, mode='RGB')
            #img = misc.imresize(img, (orig_w,orig_h), interp='bilinear')
            img = np.array(Image.fromarray(img).resize((orig_w,orig_h),resample=Image.BILINEAR))
            w1,h1=128,128
            dw=(orig_w-w1)//2
            dh=(orig_h-h1)//2
            box = (dw, dh, orig_w-dw, orig_h-dh)
            img=img[dh:-dh,dw:-dw]
        else:
            img = imread(img_filepath)#, mode='RGB')
        
        #x = misc.imresize(img, (self.w,self.h), interp='bilinear').astype(float)
        x = np.array(Image.fromarray(img).resize((self.w,self.h),resample=Image.BILINEAR)).astype(float)
        
        if self.convert2BGR:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            # Zero-center by mean pixel
            if self.imageNetUtilsMean: #imagenet.utils caffe
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.68
            else: #vggface-2
                x[..., 0] -= 91.4953
                x[..., 1] -= 103.8827
                x[..., 2] -= 131.0912
        else:
            #x=(x-127.5)/128.0
            x /= 127.5
            x -= 1.
            #x=x/128.0-1.0
        return x
        
    def extract_features(self,img_filepath,crop_center=False):
        x=self.preprocess_image(img_filepath,crop_center)
        x = np.expand_dims(x, axis=0)
        feed_dict={self.tf_input_image: x}
        if self.tf_learning_phase is not None:
            feed_dict[self.tf_learning_phase]=self.additional_input_value
        preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).reshape(-1)
        #preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).mean(axis=(0,1,2)).reshape(-1)
        return preds
    
    def close_session(self):
        self.tf_sess.close()



def extract_insightface_features(model,img_filepath):
    img = cv2.imread(img_filepath)
    img=img[:, :, ::-1] #bgr2rgb
    embeddings = model.get_feat(img)
    if embeddings is None:
        print(img_filepath)
    #print(embeddings.shape)
    return embeddings[0]


def extract_torch_features(model, files,crop_center=False):
    if MY_TORCH_MODEL:
        from torchvision import transforms
        test_transforms = transforms.Compose(
            [
                transforms.Resize((224,224)),
                #transforms.Resize((260,260)),
                #transforms.Resize((160,160)),
                #transforms.Resize((196,196)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            ]
        )
    else:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        def test_transforms(img):
            img = np.array(img.resize((224,224),Image.BILINEAR))
            img = img[:, :, ::-1]  # RGB -> BGR
            img = img.astype(np.float32)
            img -= mean_bgr
            img = img.transpose(2, 0, 1)  # C x H x W
            img = torch.from_numpy(img).float()
            return img
    X=[]
    imgs=[]
    for filepath in files:
        img = Image.open(os.path.join(DATASET_PATH,filepath))
        if crop_center:
            orig_w,orig_h=250,250
            img = img.resize((orig_w,orig_h),Image.BILINEAR)
            w1,h1=224,224#128,128
            dw=(orig_w-w1)/2
            dh=(orig_h-h1)/2
            box = (dw, dh, orig_w-dw, orig_h-dh)
            img = img.crop(box)
        #img = img.resize((224,224),Image.BILINEAR)
        img_tensor = test_transforms(img)
            
        if img.size:
            imgs.append(img_tensor)
            if len(imgs)>=8: #1:#8:
                scores = model(torch.stack(imgs, dim=0).to('cuda'))
                scores=scores.data.cpu().numpy()
                #print(scores.shape)
                if len(X)==0:
                    X=scores
                else:
                    X=np.concatenate((X,scores),axis=0)
                
                imgs=[]

    if len(imgs)>0:        
        scores = model(torch.stack(imgs, dim=0).to('cuda'))
        scores=scores.data.cpu().numpy()
        #print(scores.shape)
        if len(X)==0:
            X=scores
        else:
            X=np.concatenate((X,scores),axis=0)
    
    return X
    
def chi2dist(x, y):
    sum=x+y
    chi2array = np.divide((x-y)**2, sum, out=np.zeros_like(x), where=sum>0)
    #chi2array=np.where(sum>0, (x-y)**2/sum, 0)
    return np.sum(chi2array)

def KL_dist(x, y):
    KL_array=(x+0.001)*np.log((x+0.001)/(y+0.001))
    return np.sum(KL_array)

    
def get_single_image_per_class_cv(y, n_splits=10,random_state=0):
    res_cv=[]
    inds = np.arange(len(y))
    np.random.seed(random_state)
    for _ in range(n_splits):
        inds_train, inds_test = [], []

        for lbl in np.unique(y):
            tmp_inds = inds[y == lbl]
            np.random.shuffle(tmp_inds)
            last_ind=1
            #last_ind=math.ceil(len(tmp_inds)/2)
            if last_ind==0 and len(tmp_inds)>0:
                last_ind=1
            inds_train.extend(tmp_inds[:last_ind])
            inds_test.extend(tmp_inds[last_ind:])
            
        inds_train = np.array(inds_train)
        inds_test = np.array(inds_test)
    
        res_cv.append((inds_train, inds_test))
    return res_cv

def classifier_tester(classifier,x,y):
    sss=get_single_image_per_class_cv(y)
    #sss=model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    scores=model_selection.cross_validate(classifier,x, y, scoring='accuracy',cv=sss)
    acc=scores['test_score']
    print('accuracies=',acc*100)
    print('total acc=',round(acc.mean()*100,2),round(acc.std()*100,2))
    print('test time=',scores['score_time'])


def get_cnn_model():
    if use_framework==INSIGHTFACE:
        import insightface
        #model_path='C:/Users/avsavchenko/.insightface/models/buffalo_l/w600k_r50.onnx'
        model_path='D:/src_code/DNN_models/age_gender/insightface/models/vgg2_r50_pfc.onnx'
        cnn_model = insightface.model_zoo.get_model(model_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        cnn_model.prepare(ctx_id=0)
        print(cnn_model)
    elif use_framework==TORCH:
        import torch
        import torch.nn as nn            
        import timm

        if MY_TORCH_MODEL:
            if OFA_MODEL:
                if False:
                    cnn_model = OFAMobileNetV3(n_classes=9131,dropout_rate=0.,width_mult=1.2, ks_list= [3, 5, 7],expand_ratio_list=[3, 4, 6],depth_list=[2, 3, 4])
                    state_dict = torch.load('../models/ofa_mbv3_model_best_new.pth.tar', map_location="cpu")["state_dict"]
                    cnn_model.load_state_dict(state_dict)
                    #print(cnn_model)
                else:
                    model_name='subnet_device_865_acc_98.3_lut_12.1ms_w12_d234_nac_gbdt'
                    #model_name='subnet_device_865_acc_97.8_lut_9.15ms_w12_d234_nac_gbdt'
                    #model_name='subnet_device_765_acc_98.3_lut_34.6ms_w12_d234_nac_gbdt'
                    #model_name='subnet_device_765_acc_97.7_lut_24.45ms_w12_d234_nac_gbdt'

                    #model_name='subnet_device_865_acc_97.9_lut_12.1ms_w12_d234_reg_predictor'
                    #model_name='subnet_device_865_acc_97.7_lut_9.15ms_w12_d234_reg_predictor'
                    #model_name='subnet_device_765_acc_97.9_lut_34.6ms_w12_d234_reg_predictor'
                    #model_name='subnet_device_765_acc_97.7_lut_24.45ms_w12_d234_reg_predictor'
                    
                    #model_name='subnet_device_865_acc_98.3_lut_12.1ms_w12_d234_reg_gbdt'
                    #model_name='subnet_device_865_acc_97.8_lut_9.15ms_w12_d234_reg_gbdt'
                    #model_name='subnet_device_765_acc_98.4_lut_34.6ms_w12_d234_reg_gbdt'
                    #model_name='subnet_device_765_acc_97.7_lut_24.45ms_w12_d234_reg_gbdt'
                    
                    cnn_model=get_model_artifacts('../models/ofa_subnets',model_name)
                    #cnn_model=get_model_artifacts('D:/src_code/DNN_models/ofa_subnets',model_name)
                    print(model_name)
                cnn_model.classifier=torch.nn.Identity()
            elif True:
                cnn_model=timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
                cnn_model.classifier=torch.nn.Identity()
                cnn_model.load_state_dict(torch.load('D:/src_code/DNN_models/state_vggface2_enet0_new.pt')) #_new
                print('enet0')
            elif True:
                cnn_model = torch.load('D:/src_code/DNN_models/vggface2_enet25.pt')
                #cnn_model = torch.load('D:/src_code/DNN_models/vggface2_mobilenetv2.pt')
                cnn_model.classifier=torch.nn.Identity()
                print('enet25')
        else:
            cnn_model = torch.load('D:/src_code/DNN_models/vggface_senet50_1.pt')
            print('senet')
        cnn_model=cnn_model.to('cuda')
        cnn_model.eval()
    else:
        #cnn_model=TensorFlowInference('D:/src_code/HSE_FaceRec_tf/age_gender_identity/age_gender_tf2_new-01-0.14-0.92.pb',input_tensor='input_1:0',output_tensor='global_pooling/Mean:0',convert2BGR=True, imageNetUtilsMean=True)
        cnn_model=TensorFlowInference('D:/src_code/DNN_models/facenet_inceptionresnet/20180402-114759.pb',input_tensor='input:0',output_tensor='embeddings:0',learning_phase_tensor='phase_train:0',convert2BGR=False) #embeddings, InceptionResnetV1/Repeat_2/block8_5/Relu, InceptionResnetV1/Repeat_1/block17_10/Relu

    return cnn_model

def compute_features(cnn_model,files):
    crop_center=False
    start_time = time.time()
    if use_framework==INSIGHTFACE:
        X=np.array([extract_insightface_features(cnn_model,os.path.join(DATASET_PATH,filepath)) for filepath in files])
    elif use_framework==TORCH:
        X=extract_torch_features(cnn_model,files)
    else:
        X=np.array([cnn_model.extract_features(os.path.join(DATASET_PATH,filepath),crop_center=crop_center) for filepath in files])
        cnn_model.close_session()
    print('--- %s seconds ---' % (time.time() - start_time))

    return X

def test_lfw_recognition():
    features_file='tmp.npz'
    save_video_features=False
    if not os.path.exists(features_file) or save_video_features:
        print("Creating file ",features_file)
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

        cnn_model=get_cnn_model()
        if False:
            dirs_and_files=np.array(get_files(DATASET_PATH))
        else: #LFW and YTF concatenation
            subjects = (line.rstrip('\n') for line in open('lfw_ytf_classes.txt'))
            dirs_and_files=np.array([[d,os.path.join(d,f)] for d in subjects for f in next(os.walk(os.path.join(DATASET_PATH,d)))[2] if is_image(f)])
            
        dirs=dirs_and_files[:,0]
        files=dirs_and_files[:,1]

        label_enc=preprocessing.LabelEncoder()
        label_enc.fit(dirs)
        y=label_enc.transform(dirs)
        X=compute_features(cnn_model,files)
        print ('X.shape=',X.shape)
        #print ('X[0,5]=',X[:,0:6])
        #np.savez(features_file,x=X,y=y)
    else:
        data = np.load(features_file)
        X=data['x']
        y=data['y']
    #if len(X.shape)>2:
    #    print('orig shape:',X.shape)
    #    X=np.reshape(X,X.shape[:2])
    #    print('new shape:',X.shape)
    #X=X-X.mean(axis=1, keepdims=True)
    
    X_norm=preprocessing.normalize(X,norm='l2')
    
    y_l=list(y)
    indices=[i for i,el in enumerate(y_l) if y_l.count(el) > 1]
    y=y[indices]
    label_enc=preprocessing.LabelEncoder()
    label_enc.fit(y)
    y=label_enc.transform(y)
    X_norm=X_norm[indices,:]
    print('after loading: num_classes=',len(label_enc.classes_),' X shape:',X.shape,' X_norm shape:',X_norm.shape)
    if True:
        pca_components=128 #256
        classifiers=[]
        #classifiers.append(['lightGBM',LGBMClassifier(max_depth=3,n_estimators=200)])
        #classifiers.append(['xgboost',XGBClassifier(max_depth=3,n_estimators=200)])
        classifiers.append(['k-NN+PCA',Pipeline(steps=[('pca', PCA(n_components=pca_components)), ('classifier', KNeighborsClassifier(n_neighbors=1,p=2))])])
        classifiers.append(['k-NN',KNeighborsClassifier(n_neighbors=1,p=2)])
        #classifiers.append(['k-NN chisq',KNeighborsClassifier(n_neighbors=1,metric=chi2dist)])
        #classifiers.append(['k-NN KL',KNeighborsClassifier(n_neighbors=1,metric=KL_dist)])
        #classifiers.append(['rf',RandomForestClassifier(n_estimators=100,max_depth=10)])
        #classifiers.append(['svm',SVC()])
        #classifiers.append(['linear svm',LinearSVC()])
        for cls_name,classifier in classifiers:
            print(cls_name)
            classifier_tester(classifier,X_norm,y)
    else:
        classifier=KNeighborsClassifier(1)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X_norm, y, test_size=0.5, random_state=42, stratify=y)
        print (X_train.shape,X_test.shape)
        print(y_train.shape,y_test.shape)
        print('train classes:',len(np.unique(y_train)))
        classifier.fit(X_train,y_train)
        y_test_pred=classifier.predict(X_test)
        acc=100.0*(y_test==y_test_pred).sum()/len(y_test)
        print('acc=',acc)


import lfw
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate


def test_lfw_verification():
    #pairs=loadPairs('D:/datasets/lfw_ytf/pairs.txt')
    pairs = lfw.read_pairs('D:/datasets/lfw_ytf/pairs.txt')
    #print(pairs.shape,pairs[:5],pairs[-5:])
    # Get the paths for the corresponding images
    paths, actual_issame = lfw.get_paths(DATASET_PATH, pairs)
    #print(len(paths), len(actual_issame), paths[:5], actual_issame[:5], paths[-5:], actual_issame[-5:])
    
    embedding_indices=np.zeros(len(paths), dtype=int)
    filepaths={}
    for i,p in enumerate(paths):
        if p not in filepaths:
            filepaths[p]=len(filepaths)
        embedding_indices[i]=filepaths[p]
    #print(len(filepaths),len(embedding_indices), len(actual_issame), embedding_indices[:5], actual_issame[:5], embedding_indices[-5:], actual_issame[-5:])
    
    files=[None]*len(filepaths)
    for f,i in filepaths.items():
        files[i]=f
    #print(files[:5],files[-5:])
    
    features_file='tmp.npz'
    if not os.path.exists(features_file) or True:
        cnn_model=get_cnn_model()
        X=compute_features(cnn_model,files)
        #np.savez(features_file,x=X)
    else:
        data = np.load(features_file)
        X=data['x']
    #print(X.shape)
    if True:
        X=preprocessing.normalize(X,norm='l2')
        

    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    embedding_size=X.shape[1]
    embeddings = np.zeros((nrof_embeddings, embedding_size))
    for i,emb_ind in enumerate(embedding_indices):
        embeddings[i,:]=X[emb_ind]

    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0)
    
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.5f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
    

if __name__ == '__main__':
    if True:
        test_lfw_recognition()
    else:
        test_lfw_verification()
    
    sys.exit(0)
        
