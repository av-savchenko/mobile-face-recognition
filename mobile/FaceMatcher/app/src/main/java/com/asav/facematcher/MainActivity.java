package com.asav.facematcher;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.*;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.view.Display;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.*;

import com.asav.facematcher.mtcnn.Box;
import com.asav.facematcher.mtcnn.MTCNN;

import org.opencv.android.*;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.gpu.CompatibilityList;

import java.io.*;
import java.util.*;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;

    private ImageView imageView=null;
    private Mat sampledImage=null;
    private static int minFaceSize=40;
    private MTCNN mtcnnFaceDetector=null;

    private List<DeepModel> deepModels =new ArrayList<>();
    private Random rnd=new Random();

    private TextView textView;
    private Spinner modelsSpinner;
    private Integer numAttempts = 100;
    private Integer numStartAttempts = 10;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        androidx.appcompat.widget.Toolbar toolbar = (androidx.appcompat.widget.Toolbar) findViewById(R.id.main_toolbar);
        setSupportActionBar(toolbar);

        imageView=(ImageView)findViewById(R.id.inputImageView);
        textView = findViewById(R.id.text);
        textView.setMovementMethod(new ScrollingMovementMethod());
        modelsSpinner=findViewById(R.id.models_spinner);

        try {
            mtcnnFaceDetector = MTCNN.Companion.create(getAssets()); //new MTCNN(this);
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing MTCNNModel!"+e);
        }

        CompatibilityList compatList = new CompatibilityList();
        boolean hasGPU=compatList.isDelegateSupportedOnThisDevice();
        boolean needCPU=true;//!hasGPU;
        Log.i(TAG, "hasGPU " + hasGPU);

        List<String> modelNames=new ArrayList<>();
        try {
            for (String asset : getAssets().list("")){
                if (asset.startsWith("mtcnn_"))
                    continue;
                try{
                    if(asset.toLowerCase().endsWith(".tflite")){
                        String modelName=asset.substring(0,asset.length()-7);
                        if(needCPU) {
                            deepModels.add(new TfLiteFacialFeatureExtractor(this, asset, false));
                            modelNames.add(modelName + " (TfLite), CPU");
                        }
                        if(hasGPU){
                            deepModels.add(new TfLiteFacialFeatureExtractor(this, asset, true));
                            modelNames.add(modelName+" (TfLite), GPU");
                        }
                    }
                    else if(asset.toLowerCase().endsWith(".ptl")){
                        deepModels.add(new TorchLiteFacialFeatureExtractor(this,asset));
                        String modelName=asset.substring(0,asset.length()-4);
                        modelNames.add(modelName+" (Torch)");
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Error creating model: " + e+" "+Log.getStackTraceString(e));
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Error reading assets: " + e+" "+Log.getStackTraceString(e));
        }
        String modelNamesArr[]=new String[modelNames.size()];
        modelNames.toArray(modelNamesArr);
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,R.layout.spinner_item, modelNamesArr);
        modelsSpinner.setAdapter(adapter);

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.toolbar_menu, menu);
        return true;
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                    Toast.makeText(getApplicationContext(),
                            "OpenCV error",
                            Toast.LENGTH_SHORT).show();
                } break;
            }
        }
    };
    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    getPackageManager()
                            .getPackageInfo(getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }
    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            int status= ContextCompat.checkSelfPermission(this,permission);
            if (ContextCompat.checkSelfPermission(this,permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }



    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS:
                Map<String, Integer> perms = new HashMap<String, Integer>();
                boolean allGranted = true;
                for (int i = 0; i < permissions.length; i++) {
                    perms.put(permissions[i], grantResults[i]);
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED)
                        allGranted = false;
                }
                // Check for ACCESS_FINE_LOCATION
                if (!allGranted) {
                    // Permission Denied
                    Toast.makeText(MainActivity.this, "Some Permission is Denied", Toast.LENGTH_SHORT)
                            .show();
                    finish();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    private static final int SELECT_PICTURE = 1;
    private static final int SELECT_TEMPLATE_PICTURE_MATCH = 2;
    private void openImageFile(int requestCode){
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent,"Select Picture"),requestCode);
    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_openGallery:
                openImageFile(SELECT_PICTURE);
                return true;
            case R.id.action_matchfaces:
                if(isImageLoaded()) {
                    openImageFile(SELECT_TEMPLATE_PICTURE_MATCH);
                }
                return true;
            case R.id.action_computeRunningTime:
                action_computeRunningTime();
                return true;
            default:
                // If we got here, the user's action was not recognized.
                // Invoke the superclass to handle it.
                return super.onOptionsItemSelected(item);
        }
    }
    private boolean isImageLoaded(){
        if(sampledImage==null)
            Toast.makeText(getApplicationContext(),
                    "It is necessary to open image firstly",
                    Toast.LENGTH_SHORT).show();
        return sampledImage!=null;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode==RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Log.d(TAG, "uri" + selectedImageUri);
                sampledImage=convertToMat(selectedImageUri);
                if(sampledImage!=null)
                    displayImage(sampledImage);
            }
            else if(requestCode==SELECT_TEMPLATE_PICTURE_MATCH){
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Mat imageToMatch=convertToMat(selectedImageUri);
                matchFaces(sampledImage,imageToMatch);
            }
        }
    }
    private Mat convertToMat(Uri selectedImageUri)
    {
        Mat resImage=null;
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            Bitmap bmp= BitmapFactory.decodeStream(ims);
            Mat rgbImage=new Mat();
            Utils.bitmapToMat(bmp, rgbImage);
            ims.close();
            ims = getContentResolver().openInputStream(selectedImageUri);
            ExifInterface exif = new ExifInterface(ims);//selectedImageUri.getPath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,1);
            switch (orientation)
            {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    //get the mirrored image
                    rgbImage=rgbImage.t();
                    //flip on the y-axis
                    Core.flip(rgbImage, rgbImage, 1);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    //get up side down image
                    rgbImage=rgbImage.t();
                    //Flip on the x-axis
                    Core.flip(rgbImage, rgbImage, 0);
                    break;
            }

            Display display = getWindowManager().getDefaultDisplay();
            android.graphics.Point size = new android.graphics.Point();
            display.getSize(size);
            int width = size.x;
            int height = size.y;
            double downSampleRatio= calculateSubSampleSize(rgbImage,width,height);
            resImage=new Mat();
            Imgproc.resize(rgbImage, resImage, new
                    Size(),downSampleRatio,downSampleRatio,Imgproc.INTER_AREA);
        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
            resImage=null;
        }
        return resImage;
    }
    private static double calculateSubSampleSize(Mat srcImage, int reqWidth,
                                                 int reqHeight) {
        final int height = srcImage.height();
        final int width = srcImage.width();
        double inSampleSize = 1;
        if (height > reqHeight || width > reqWidth) {
            final double heightRatio = (double) reqHeight / (double) height;
            final double widthRatio = (double) reqWidth / (double) width;
            inSampleSize = heightRatio<widthRatio ? heightRatio :widthRatio;
        }
        return inSampleSize;
    }
    private void displayImage(Mat image)
    {
        Bitmap bitmap = Bitmap.createBitmap(image.cols(),
                image.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(image, bitmap);
        imageView.setImageBitmap(bitmap);
    }


    private void matchFaces(Mat img1, Mat img2){
        Mat resImage = new Mat();
        if(img2.rows()!=img1.rows()){
            Imgproc.resize(img2,img2,img1.size());
        }
        List<Mat> src = Arrays.asList(img1, img2);
        Core.hconcat(src, resImage);
        List<FaceFeatures> features1=getFacesFeatures(img1);
        List<FaceFeatures> features2=getFacesFeatures(img2);
        for(FaceFeatures face1 : features1){
            double minDist=10000;
            FaceFeatures bestFace=null;
            for(FaceFeatures face2 : features2){
                double dist = 0;
                for (int i = 0; i < face1.features.length; ++i) {
                    dist += (face1.features[i] - face2.features[i]) * (face1.features[i] - face2.features[i]);
                }
                dist = Math.sqrt(dist);
                if(dist<minDist){
                    minDist=dist;
                    bestFace=face2;
                }
            }
            if(bestFace!=null && minDist<1){
                float x=bestFace.centerX;
                Imgproc.line(resImage,new Point(face1.centerX*img1.cols(),face1.centerY*img1.rows()),
                        new Point(img1.cols()+bestFace.centerX*img2.cols(),bestFace.centerY*img2.rows()),
                        new Scalar(255,0,0),5);
                Log.i(TAG,"distance "+minDist);
            }
        }
        displayImage(resImage);
    }
    private List<FaceFeatures> getFacesFeatures(Mat img){
        DeepModel featureExtractor= deepModels.get(modelsSpinner.getSelectedItemPosition());
        Bitmap bmp = Bitmap.createBitmap(img.cols(), img.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(img, bmp);

        Bitmap resizedBitmap=bmp;
        double minSize=600.0;
        double scale=Math.min(bmp.getWidth(),bmp.getHeight())/minSize;
        if(scale>1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(bmp, (int)(bmp.getWidth()/scale), (int)(bmp.getHeight()/scale), false);
            //bmp=resizedBitmap;
        }
        long startTime = SystemClock.uptimeMillis();
        Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(TAG, "Timecost to run mtcnn: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        List<FaceFeatures> facesInfo=new ArrayList<>();
        for (Box box : bboxes) {
            android.graphics.Rect bbox = new android.graphics.Rect(Math.max(0,bmp.getWidth()*box.left() / resizedBitmap.getWidth()),
                    Math.max(0,bmp.getHeight()* box.top() / resizedBitmap.getHeight()),
                    bmp.getWidth()* box.right() / resizedBitmap.getWidth(),
                    bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
            );
            Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
            FacialEmbeddings res=featureExtractor.processImage(faceBitmap);
            facesInfo.add(new FaceFeatures(res.features,0.5f*(box.left()+box.right()) / resizedBitmap.getWidth(),0.5f*(box.top()+box.bottom()) / resizedBitmap.getHeight()));
        }
        return facesInfo;
    }
    private class FaceFeatures{
        public FaceFeatures(float[] feat, float x, float y){
            features=feat;
            centerX=x;
            centerY=y;
        }
        public float[] features;
        public float centerX,centerY;
    }

    private void action_computeRunningTime(){
        Pair<Double,Double> res= deepModels.get(modelsSpinner.getSelectedItemPosition()).testPerformance(numAttempts,numStartAttempts);
        textView.setText(String.format("%s mean=%.3f std=%.3f",modelsSpinner.getSelectedItem(),res.first,res.second));
    }

}