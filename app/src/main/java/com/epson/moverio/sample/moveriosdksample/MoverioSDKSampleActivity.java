/*
 * Copyright(C) Seiko Epson Corporation 2018. All rights reserved.
 *
 * Warranty Disclaimers.
 * You acknowledge and agree that the use of the software is at your own risk.
 * The software is provided "as is" and without any warranty of any kind.
 * Epson and its licensors do not and cannot warrant the performance or results
 * you may obtain by using the software.
 * Epson and its licensors make no warranties, express or implied, as to non-infringement,
 * merchantability or fitness for any particular purpose.
 */

package com.epson.moverio.sample.moveriosdksample;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.LinearLayout;

public class MoverioSDKSampleActivity extends Activity {
    private final String TAG = this.getClass().getSimpleName();

    private Context mContext = null;

    private LinearLayout mLinearLayout_display = null;
    private LinearLayout mLinearLayout_sensor = null;
    private LinearLayout mLinearLayout_camera = null;;
    private LinearLayout mLinearLayout_audio = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_moverio_sdk_sample);
        mContext = this;

        mLinearLayout_display = (LinearLayout) findViewById(R.id.linearLayout_display_sample);
        mLinearLayout_display.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(mContext, MoverioDisplaySampleActivity.class);
                startActivity(intent);
            }
        });
        mLinearLayout_sensor = (LinearLayout) findViewById(R.id.linearLayout_sensor_sample);
        mLinearLayout_sensor.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(mContext, MoverioSensorSampleActivity.class);
                startActivity(intent);
            }
        });
        mLinearLayout_camera = (LinearLayout) findViewById(R.id.linearLayout_camera_sample);
        mLinearLayout_camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(mContext, MoverioCameraSampleActivity.class);
                startActivity(intent);
            }
        });
        mLinearLayout_audio = (LinearLayout) findViewById(R.id.linearLayout_audio_sample);
        mLinearLayout_audio.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(mContext, MoverioAudioSampleActivity.class);
                startActivity(intent);
            }
        });
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }
}
