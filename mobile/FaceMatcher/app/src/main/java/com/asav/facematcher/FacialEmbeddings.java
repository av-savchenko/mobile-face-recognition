package com.asav.facematcher;

import java.io.Serializable;


/**
 * Created by avsavchenko.
 */
public class FacialEmbeddings{
    public float[] features=null;

    public FacialEmbeddings(float[] features){
        this.features=new float[features.length];
        float sum = 0;
        for (int i = 0; i < features.length; ++i) {
            sum += features[i] * features[i];
        }
        sum = (float) Math.sqrt(sum);
        if (sum<0.00001)
            sum=1;
        for (int i = 0; i < features.length; ++i)
            this.features[i]=features[i] / sum;
    }
}
