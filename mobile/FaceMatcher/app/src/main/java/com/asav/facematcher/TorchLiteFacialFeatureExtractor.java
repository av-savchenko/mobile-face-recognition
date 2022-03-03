package com.asav.facematcher;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;
import org.pytorch.Device;
import org.pytorch.IValue;
import org.pytorch.Tensor;
import org.pytorch.Module;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.LiteModuleLoader;

/**
 * Created by avsavchenko.
 */
public class TorchLiteFacialFeatureExtractor implements DeepModel{

    /** Tag for the {@link Log}. */
    private static final String TAG = "TorchLiteFacialFeatureExtractor";

    private Module module=null;
    private int width=224;
    private int height=224;
    private int channels=3;

    public float[] mean_rgb, std_rgb;
    private Random rnd=new Random();

    public TorchLiteFacialFeatureExtractor(final Context context, String assetName) throws IOException {
        String model_path=assetFilePath(context,assetName);
        module=LiteModuleLoader.load(model_path,null, Device.CPU);
        if(assetName.startsWith("PocketNet")){
            mean_rgb = new float[] {0.5f, 0.5f, 0.5f};
            std_rgb = new float[] {0.5f, 0.5f, 0.5f};
        }
        else{
            mean_rgb = TensorImageUtils.TORCHVISION_NORM_MEAN_RGB;
            std_rgb = TensorImageUtils.TORCHVISION_NORM_STD_RGB;
        }
    }
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private static MappedByteBuffer loadModelFile(Context context, String modelFile) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        MappedByteBuffer retFile = inputStream.getChannel().map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
        fileDescriptor.close();
        return retFile;
    }
    public FacialEmbeddings processImage(Bitmap bitmap) {
        bitmap=Bitmap.createScaledBitmap(bitmap, width, height, false);
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,mean_rgb,std_rgb);
        long startTime = SystemClock.uptimeMillis();
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        long timecostMs=SystemClock.uptimeMillis() - startTime;
        Log.i(TAG, "Timecost to run model inference: " + timecostMs);
        final float[] features = outputTensor.getDataAsFloatArray();
        Log.i(TAG, "!!!!!!!!!!!!!!!!!!!!!!!!! end feature extraction first feat=" + features[0]
                + " last feat=" + features[features.length - 1]+" total lenghth="+features.length);
        FacialEmbeddings res=new FacialEmbeddings(features);
        return res;
    }
    public Pair<Double,Double> testPerformance(int num_attempts, int num_start_attempts){
        int numElements=channels * width * height;
        final java.nio.FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(numElements);
        for (int i = 0; i < numElements; i++) {
            floatBuffer.put(i, 2*rnd.nextFloat()-1);
        }
        final Tensor inputTensor = Tensor.fromBlob(floatBuffer, new long[] {1, channels, height, width});

        for(int i=0;i<num_start_attempts;++i){
            final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        }

        double meanTime=0;
        double stdTime=0;
        for(int a=0;a<num_attempts;++a){
            long startTime = SystemClock.uptimeMillis();
            final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
            long duration=(SystemClock.uptimeMillis() - startTime);
            //Log.i(TAG, "Pytorch timecost to run model inference: " + duration+" at iter="+a);
            meanTime+=duration;
            stdTime+=duration*duration;
        }
        meanTime/=num_attempts;
        stdTime=Math.sqrt((stdTime/num_attempts)-meanTime*meanTime);
        return new Pair<Double,Double>(meanTime,stdTime);
    }
}
