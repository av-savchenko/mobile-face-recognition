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
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;
import org.pytorch.Device;
import org.pytorch.IValue;
import org.pytorch.MemoryFormat;
import org.pytorch.Tensor;
import org.pytorch.Module;
//import org.pytorch.torchvision.TensorImageUtils;
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
            mean_rgb = new float[] {0.485f, 0.456f, 0.406f};
            std_rgb = new float[] {0.229f, 0.224f, 0.225f};
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
    /**
     * Writes tensor content from specified {@link android.graphics.Bitmap}, normalized with specified
     * in parameters mean and std to specified {@link java.nio.FloatBuffer} with specified offset.
     *
     * @param bitmap {@link android.graphics.Bitmap} as a source for Tensor data
     * @param x - x coordinate of top left corner of bitmap's area
     * @param y - y coordinate of top left corner of bitmap's area
     * @param width - width of bitmap's area
     * @param height - height of bitmap's area
     * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
     * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
     *     order
     */
    public static void bitmapToFloatBuffer(
            final Bitmap bitmap,
            final int x,
            final int y,
            final int width,
            final int height,
            final float[] normMeanRGB,
            final float[] normStdRGB,
            final FloatBuffer outBuffer,
            final int outBufferOffset,
            final MemoryFormat memoryFormat) {
         if (memoryFormat != MemoryFormat.CONTIGUOUS && memoryFormat != MemoryFormat.CHANNELS_LAST) {
            throw new IllegalArgumentException("Unsupported memory format " + memoryFormat);
        }

        final int pixelsCount = height * width;
        final int[] pixels = new int[pixelsCount];
        bitmap.getPixels(pixels, 0, width, x, y, width, height);
        if (MemoryFormat.CONTIGUOUS == memoryFormat) {
            final int offset_g = pixelsCount;
            final int offset_b = 2 * pixelsCount;
            for (int i = 0; i < pixelsCount; i++) {
                final int c = pixels[i];
                float r = ((c >> 16) & 0xff) / 255.0f;
                float g = ((c >> 8) & 0xff) / 255.0f;
                float b = ((c) & 0xff) / 255.0f;
                outBuffer.put(outBufferOffset + i, (r - normMeanRGB[0]) / normStdRGB[0]);
                outBuffer.put(outBufferOffset + offset_g + i, (g - normMeanRGB[1]) / normStdRGB[1]);
                outBuffer.put(outBufferOffset + offset_b + i, (b - normMeanRGB[2]) / normStdRGB[2]);
            }
        } else {
            for (int i = 0; i < pixelsCount; i++) {
                final int c = pixels[i];
                float r = ((c >> 16) & 0xff) / 255.0f;
                float g = ((c >> 8) & 0xff) / 255.0f;
                float b = ((c) & 0xff) / 255.0f;
                outBuffer.put(outBufferOffset + 3 * i + 0, (r - normMeanRGB[0]) / normStdRGB[0]);
                outBuffer.put(outBufferOffset + 3 * i + 1, (g - normMeanRGB[1]) / normStdRGB[1]);
                outBuffer.put(outBufferOffset + 3 * i + 2, (b - normMeanRGB[2]) / normStdRGB[2]);
            }
        }
    }
    public static Tensor bitmapToFloat32Tensor(
            final Bitmap bitmap,
            int x,
            int y,
            int width,
            int height,
            float[] normMeanRGB,
            float[] normStdRGB,
            MemoryFormat memoryFormat) {
        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
        bitmapToFloatBuffer(
                bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0, memoryFormat);
        return Tensor.fromBlob(floatBuffer, new long[] {1, 3, height, width}, memoryFormat);
    }
    public FacialEmbeddings processImage(Bitmap bitmap) {
        bitmap=Bitmap.createScaledBitmap(bitmap, width, height, false);
        final Tensor inputTensor = bitmapToFloat32Tensor(bitmap,0,
                0,
                bitmap.getWidth(),
                bitmap.getHeight(),
                mean_rgb,std_rgb,
                MemoryFormat.CONTIGUOUS);
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
