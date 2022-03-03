package com.asav.facematcher;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.common.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Created by avsavchenko.
 */
public abstract class TfLiteClassifier  implements DeepModel{

    /** Tag for the {@link Log}. */
    private static final String TAG = "TfLiteClassifier";

    private Random rnd=new Random();

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /* Preallocated buffers for storing image data in. */
    private int[] intValues = null;
    protected ByteBuffer imgData = null;
    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private int imageSizeX=224,imageSizeY=224, numChannels=3;
    private boolean isWHC=true;
    private float[][][] outputs;
    Map<Integer, Object> outputMap = new HashMap<>();

    public TfLiteClassifier(final Context context, final String model_path, boolean useGPU) throws IOException {
        Interpreter.Options options = (new Interpreter.Options()).setNumThreads(4);//.addDelegate(delegate);
        if (useGPU) {
            org.tensorflow.lite.gpu.GpuDelegate.Options opt=new org.tensorflow.lite.gpu.GpuDelegate.Options();
            opt.setInferencePreference(org.tensorflow.lite.gpu.GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
            org.tensorflow.lite.gpu.GpuDelegate delegate = new org.tensorflow.lite.gpu.GpuDelegate(opt);
            options.addDelegate(delegate);
        }

        MappedByteBuffer tfliteModel= FileUtil.loadMappedFile(context,model_path);
        tflite = new Interpreter(tfliteModel,options);
        tflite.allocateTensors();
        int[] inputShape=tflite.getInputTensor(0).shape();
        if (inputShape[3]<=3) {
            imageSizeX = inputShape[1];
            imageSizeY = inputShape[2];
            numChannels = inputShape[3];
            isWHC=true;
        }
        else
        {
            imageSizeX = inputShape[2];
            imageSizeY = inputShape[3];
            numChannels = inputShape[1];
            isWHC=false;
        }
        Log.i(TAG, "Model " + model_path+" "+imageSizeX+" "+imageSizeY+" "+numChannels+" isWHC "+isWHC);
        intValues = new int[imageSizeX * imageSizeY];
        imgData =ByteBuffer.allocateDirect(imageSizeX*imageSizeY* numChannels*getNumBytesPerChannel());
        imgData.order(ByteOrder.nativeOrder());

        int outputCount=tflite.getOutputTensorCount();
        outputs=new float[outputCount][1][];
        for(int i = 0; i< outputCount; ++i) {
            int[] shape=tflite.getOutputTensor(i).shape();
            int numOFFeatures = shape[1];
            Log.i(TAG, "Read output layer size is " + numOFFeatures);
            outputs[i][0] = new float[numOFFeatures];
            ByteBuffer ith_output = ByteBuffer.allocateDirect( numOFFeatures* getNumBytesPerChannel());  // Float tensor, shape 3x2x4
            ith_output.order(ByteOrder.nativeOrder());
            outputMap.put(i, ith_output);
        }
    }

    /** Classifies a frame from the preview stream. */
    public FacialEmbeddings processImage(Bitmap bitmap) {
        if(bitmap.getWidth()!=imageSizeX || bitmap.getHeight()!=imageSizeY)
            bitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, false);

        Object[] inputs={null};
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        if (imgData == null) {
            return null;
        }
        imgData.rewind();
        // Convert the image to floating point.
        if(isWHC) {
            int pixel = 0;
            for (int i = 0; i < imageSizeX; ++i) {
                for (int j = 0; j < imageSizeY; ++j) {
                    final int val = intValues[pixel++];
                    for (int c = 0; c < numChannels; ++c) {
                        imgData.putFloat(getPixelValue(val, c));
                    }
                }
            }
        }
        else{
            for (int c = 0; c < numChannels; ++c) {
                int pixel = 0;
                for (int i = 0; i < imageSizeX; ++i) {
                    for (int j = 0; j < imageSizeY; ++j) {
                        final int val = intValues[pixel++];
                        imgData.putFloat(getPixelValue(val, c));
                    }
                }
            }
        }
        inputs[0] = imgData;
        long startTime = SystemClock.uptimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, outputMap);
        for(int i = 0; i< outputs.length; ++i) {
            ByteBuffer ith_output=(ByteBuffer)outputMap.get(i);
            ith_output.rewind();
            int len=outputs[i][0].length;
            for(int j=0;j<len;++j){
                outputs[i][0][j]=ith_output.getFloat();
            }
            ith_output.rewind();
        }
        long endTime = SystemClock.uptimeMillis();
        Log.i(TAG, "tf lite timecost to run model inference: " + Long.toString(endTime - startTime));

        return getResults(outputs);
    }

    public Pair<Double,Double> testPerformance(int num_attempts, int num_start_attempts){
        Object[] inputs={null};
        if (imgData == null) {
            return null;
        }
        imgData.rewind();
        //Log.d(TAG, imageSizeX + " " + imageSizeY + " " + numChannels);
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                for(int c=0;c<numChannels;++c)
                    imgData.putFloat(2 * rnd.nextFloat() - 1);
            }
        }
        inputs[0] = imgData;

        for(int a=0;a<num_start_attempts;++a) {
            tflite.runForMultipleInputsOutputs(inputs, outputMap);
            for(int i = 0; i< outputs.length; ++i) {
                ByteBuffer ith_output=(ByteBuffer)outputMap.get(i);
                ith_output.rewind();
            }
        }
        double meanTime=0;
        double stdTime=0;
        for(int a=0;a<num_attempts;++a){
            long startTime = SystemClock.uptimeMillis();
            tflite.runForMultipleInputsOutputs(inputs, outputMap);
            long duration=(SystemClock.uptimeMillis() - startTime);
            //Log.i(TAG, "Tflite timecost to run model inference: " + duration+" at iter="+a);
            meanTime+=duration;
            stdTime+=duration*duration;
            for(int i = 0; i< outputs.length; ++i) {
                ByteBuffer ith_output=(ByteBuffer)outputMap.get(i);
                ith_output.rewind();
            }
        }
        meanTime/=num_attempts;
        stdTime=Math.sqrt(stdTime/num_attempts-meanTime*meanTime);
        return new Pair<Double,Double>(meanTime,stdTime);
    }
    public void close() {
        tflite.close();
    }

    protected abstract float getPixelValue(int val, int channel);
    protected abstract FacialEmbeddings getResults(float[][][] outputs);
    public int getImageSizeX() {
        return imageSizeX;
    }
    public int getImageSizeY() {
        return imageSizeY;
    }
    protected int getNumBytesPerChannel() {
        return 4; // Float.SIZE / Byte.SIZE;
    }
}
