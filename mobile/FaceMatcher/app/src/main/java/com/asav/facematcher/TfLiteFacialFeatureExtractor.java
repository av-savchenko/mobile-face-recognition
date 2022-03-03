package com.asav.facematcher;

import android.content.Context;
import android.util.Log;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by avsavchenko.
 */
public class TfLiteFacialFeatureExtractor extends TfLiteClassifier{

    /** Tag for the {@link Log}. */
    private static final String TAG = "FacialFeatureExtractor";

    private boolean torchPreprocessing = true;
    private boolean onnxPreprocessing = false;

    public TfLiteFacialFeatureExtractor(final Context context, final String model_path, boolean useGPU) throws IOException {
        super(context,model_path,useGPU);
        torchPreprocessing=model_path.contains("subnet_device");
        onnxPreprocessing=model_path.contains("onnx") || model_path=="MobileFaceNet";
        Log.i(TAG, "Model " + model_path+" torchPreprocessing="+torchPreprocessing+" onnxPreprocessing="+onnxPreprocessing);
    }

    protected float getPixelValue(int val, int channel){
        float res=0;
        if(torchPreprocessing) {
            switch(channel){
                case 0:
                    res=(((val >> 16) & 0xFF) / 255.0f - 0.485f) / 0.229f;
                    break;
                case 1:
                    res=(((val >> 8) & 0xFF) / 255.0f - 0.456f) / 0.224f;
                    break;
                case 2:
                    res=((val & 0xFF) / 255.0f - 0.406f) / 0.225f;
                    break;
            }
        }
        else if(onnxPreprocessing){
            switch(channel){
                case 0:
                    res=((val >> 16) & 0xFF) / 127.5f - 1f;
                    break;
                case 1:
                    res=((val >> 8) & 0xFF) / 127.5f - 1f;
                    break;
                case 2:
                    res=(val & 0xFF) / 127.5f - 1f;
                    break;
            }
        }
        else{
            switch(channel){
                case 0:
                    res=(val & 0xFF) - 103.939f;
                    break;
                case 1:
                    res=((val >> 8) & 0xFF) - 116.779f;
                    break;
                case 2:
                    res=((val >> 16) & 0xFF) - 123.68f;
                    break;
            }
        }
        return res;
    }

    protected FacialEmbeddings getResults(float[][][] outputs) {
        float[] features=outputs[outputs.length-1][0];
        Log.i(TAG, "!!!!!!!!!!!!!!!!!!!!!!!!! end feature extraction first feat=" + features[0]
                + " last feat=" + features[features.length - 1]+" total lenghth="+features.length);
        FacialEmbeddings res=new FacialEmbeddings(features);
        return res;
    }
}
