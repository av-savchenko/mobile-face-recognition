package com.asav.facematcher;

import android.graphics.Bitmap;
import android.util.Pair;

public interface DeepModel {
    public FacialEmbeddings processImage(Bitmap bitmap);
    public Pair<Double,Double> testPerformance(int num_attempts, int num_start_attempts);
}
