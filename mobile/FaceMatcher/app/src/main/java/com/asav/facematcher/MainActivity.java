package com.asav.facematcher;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.*;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.*;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.media.ExifInterface;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.*;

import com.asav.facematcher.mtcnn.Box;
import com.asav.facematcher.mtcnn.MTCNN;

import org.tensorflow.lite.gpu.CompatibilityList;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;

    private ImageView imageView=null;
    private Bitmap sampledImage=null;
    private static int minFaceSize=40;
    private MTCNN mtcnnFaceDetector=null;

    private List<DeepModel> deepModels =new ArrayList<>();
    private Random rnd=new Random();

    private HandlerThread mBackgroundThread=null;
    private Handler mBackgroundHandler=null;

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
                if(!isCameraRunning()) {
                    openImageFile(SELECT_PICTURE);
                }
                return true;
            case R.id.action_matchfaces:
                if(!isCameraRunning() && isImageLoaded()) {
                    openImageFile(SELECT_TEMPLATE_PICTURE_MATCH);
                }
                return true;
            case R.id.action_computeRunningTime:
                if(!isCameraRunning()) {
                    action_computeRunningTime();
                }
                return true;
            case R.id.action_capturecamera:
                if(mBackgroundThread==null){
                    item.setTitle(R.string.action_StopCamera);
                    setupCameraX();
                }
                else{
                    item.setTitle(R.string.action_CaptureCamera);
                    stopCamera();
                }
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
    private boolean isCameraRunning(){
        if(mBackgroundThread!=null)
            Toast.makeText(getApplicationContext(),
                    "Stop camera firstly",
                    Toast.LENGTH_SHORT).show();
        return mBackgroundThread!=null;
    }
    private void setupCameraX() {
        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(CameraX.LensFacing.FRONT)
                .build();
        Preview preview = new Preview(previewConfig);
        mBackgroundThread = new HandlerThread("AnalysisThread");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());

        ImageAnalysis imageAnalysis = new ImageAnalysis(new ImageAnalysisConfig.Builder()
                .setLensFacing(CameraX.LensFacing.FRONT)
                .setCallbackHandler(mBackgroundHandler)
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .build());
        imageAnalysis.setAnalyzer(
                new ImageAnalysis.Analyzer() {
                    public void analyze(ImageProxy image, int rotationDegrees) {
                        sampledImage=imgToBitmap(image.getImage(), rotationDegrees);
                        processVideoFrame();
                    }
                }
        );

        CameraX.unbindAll();
        CameraX.bindToLifecycle(this, preview, imageAnalysis);
    }
    private Bitmap imgToBitmap(Image image, int rotationDegrees) {
        // NV21 is a plane of 8 bit Y values followed by interleaved  Cb Cr
        ByteBuffer ib = ByteBuffer.allocate(image.getHeight() * image.getWidth() * 2);

        ByteBuffer y = image.getPlanes()[0].getBuffer();
        ByteBuffer cr = image.getPlanes()[1].getBuffer();
        ByteBuffer cb = image.getPlanes()[2].getBuffer();
        ib.put(y);
        ib.put(cb);
        ib.put(cr);

        YuvImage yuvImage = new YuvImage(ib.array(),
                ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0,
                image.getWidth(), image.getHeight()), 50, out);
        byte[] imageBytes = out.toByteArray();
        Bitmap bm = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        Bitmap bitmap = bm;

        // On android the camera rotation and the screen rotation
        // are off by 90 degrees, so if you are capturing an image
        // in "portrait" orientation, you'll need to rotate the image.
        if (rotationDegrees != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotationDegrees);
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bm,
                    bm.getWidth(), bm.getHeight(), true);
            bitmap = Bitmap.createBitmap(scaledBitmap, 0, 0,
                    scaledBitmap.getWidth(), scaledBitmap.getHeight(), matrix, true);
        }
        return bitmap;
    }
    private void stopCamera() {
        CameraX.unbindAll();
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
        } catch (InterruptedException e) {
            Log.e(TAG, "Exception stoppingCamera!", e);
        }
        mBackgroundThread = null;
        mBackgroundHandler = null;
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode==RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Log.d(TAG, "uri" + selectedImageUri);
                sampledImage=getImage(selectedImageUri);
                if(sampledImage!=null)
                    displayImage(sampledImage);
            }
            else if(requestCode==SELECT_TEMPLATE_PICTURE_MATCH){
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Bitmap imageToMatch=getImage(selectedImageUri);
                matchFaces(sampledImage,imageToMatch);
            }
        }
    }
    private void displayImage(Bitmap bitmap)
    {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                imageView.setImageBitmap(bitmap);
            }
        });
    }

    private void matchFaces(Bitmap img1, Bitmap img2){
        if(img2.getHeight()!=img1.getHeight()){
            img2=Bitmap.createScaledBitmap(img2, img1.getWidth(), img1.getHeight(), false);
        }
        Bitmap resImage = Bitmap.createBitmap(img1.getWidth()+img2.getWidth(), img1.getHeight(), img1.getConfig());
        Canvas canvas = new Canvas(resImage);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStrokeWidth(5);
        canvas.drawBitmap(img1, 0f, 0f, null);
        canvas.drawBitmap(img2, img1.getWidth(), 0f, null);
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
                canvas.drawLine(face1.centerX*img1.getWidth(),face1.centerY*img1.getHeight(), img1.getWidth()+bestFace.centerX*img2.getWidth(),bestFace.centerY*img2.getHeight(), paint);
                Log.i(TAG,"distance "+minDist);
            }
        }
        displayImage(resImage);
    }
    private List<FaceFeatures> getFacesFeatures(Bitmap bmp){
        DeepModel featureExtractor= deepModels.get(modelsSpinner.getSelectedItemPosition());
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
                    Math.min(bmp.getWidth(),bmp.getWidth()* box.right() / resizedBitmap.getWidth()),
                    Math.min(bmp.getHeight(), bmp.getHeight() * box.bottom() / resizedBitmap.getHeight())
            );
            Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
            FacialEmbeddings res=featureExtractor.processImage(faceBitmap);
            facesInfo.add(new FaceFeatures(res.features,1f*box.left() / resizedBitmap.getWidth(),1f*box.top()/ resizedBitmap.getHeight(),1f*box.right() / resizedBitmap.getWidth(),1f*box.bottom() / resizedBitmap.getHeight()));
        }
        return facesInfo;
    }
    private class FaceFeatures{
        public FaceFeatures(float[] feat, float left, float top, float right, float bottom){
            features=feat;
            centerX=0.5f*(left+right);
            centerY=0.5f*(top+bottom);
            this.left=Math.max(0.f,Math.min(1f,left));
            this.top=Math.max(0.f,Math.min(1f,top));
            this.right=Math.max(0.f,Math.min(1f,right));
            this.bottom=Math.max(0.f,Math.min(1f,bottom));
        }
        public float[] features;
        public float centerX,centerY;
        public float left,top,right,bottom;
    }
    private static String[] emotions={"","Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"};

    public static String getEmotion(float[] emotionScores){
        int bestInd=-1;
        if (emotionScores!=null && emotionScores.length==emotions.length-1){
            float maxScore=0;
            for(int i=0;i<emotionScores.length;++i){
                if(maxScore<emotionScores[i]){
                    maxScore=emotionScores[i];
                    bestInd=i;
                }
            }
        }
        return emotions[bestInd+1];
    }
    private void processVideoFrame(){
        Bitmap bmp = sampledImage;
        List<FaceFeatures> features=getFacesFeatures(sampledImage);
        Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(tempBmp);
        Paint p = new Paint();
        p.setStyle(Paint.Style.STROKE);
        p.setAntiAlias(true);
        p.setFilterBitmap(true);
        p.setDither(true);
        p.setColor(Color.BLUE);
        p.setStrokeWidth(5);
        p.setColor(Color.RED);

        Paint p_text = new Paint();
        p_text.setColor(Color.WHITE);
        p_text.setStyle(Paint.Style.FILL);
        p_text.setColor(Color.BLUE);
        p_text.setTextSize(24);

        c.drawBitmap(bmp, 0, 0, null);

        for(FaceFeatures face : features){
            android.graphics.Rect bbox = new android.graphics.Rect((int)(face.left*bmp.getWidth()),(int)(face.top*bmp.getHeight()), (int)(face.right*bmp.getWidth()), (int)(face.bottom*bmp.getHeight()));
            c.drawRect(bbox, p);
            String res=getEmotion(face.features);
            c.drawText(res, Math.max(0,bbox.left), Math.max(0, bbox.top - 20), p_text);
            Log.i(TAG, res);
        }
        displayImage(tempBmp);
    }


    private void action_computeRunningTime(){
        Pair<Double,Double> res= deepModels.get(modelsSpinner.getSelectedItemPosition()).testPerformance(numAttempts,numStartAttempts);
        textView.setText(String.format("%s mean=%.3f std=%.3f",modelsSpinner.getSelectedItem(),res.first,res.second));
    }
    private Bitmap getImage(Uri selectedImageUri)
    {
        Bitmap bmp=null;
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            bmp= BitmapFactory.decodeStream(ims);
            ims.close();
            ims = getContentResolver().openInputStream(selectedImageUri);
            ExifInterface exif = new ExifInterface(ims);//selectedImageUri.getPath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,1);
            int degreesForRotation=0;
            switch (orientation)
            {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degreesForRotation=90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degreesForRotation=270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degreesForRotation=180;
                    break;
            }
            if(degreesForRotation!=0) {
                Matrix matrix = new Matrix();
                matrix.setRotate(degreesForRotation);
                bmp=Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(),
                        bmp.getHeight(), matrix, true);
            }

        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
        }
        return bmp;
    }

}