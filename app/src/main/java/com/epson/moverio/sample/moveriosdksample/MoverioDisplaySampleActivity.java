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
import android.os.Bundle;
import android.widget.CompoundButton;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.ToggleButton;

import com.epson.moverio.hardware.display.DisplayManager;

import java.io.IOException;

public class MoverioDisplaySampleActivity extends Activity {
    private final String TAG = this.getClass().getSimpleName();

    private Context mContext = null;

    private DisplayManager mDisplayManager = null;

    private ToggleButton toggleButton_deviceOpenClose = null;
    private SeekBar mSeekBar_displayBrightness = null;
    private Switch mSwitch_brightnessMode = null;
    private Switch mSwitch_displayMode = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_moverio_display_sample);

        mContext = this;

        mDisplayManager = new DisplayManager(mContext);
        toggleButton_deviceOpenClose = (ToggleButton) findViewById(R.id.toggleButton_deviceOpenClose);
        toggleButton_deviceOpenClose.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                mSeekBar_displayBrightness.setEnabled(isChecked);
                mSwitch_brightnessMode.setEnabled(isChecked);
                mSwitch_displayMode.setEnabled(isChecked);
                if (isChecked) {
                    try{
                        mDisplayManager.open();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    mDisplayManager.close();
                }
            }
        });

        // Display brightness control
        mSeekBar_displayBrightness = (SeekBar) findViewById(R.id.seekBar_displayBrightness);
        mSeekBar_displayBrightness.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                mDisplayManager.setBrightness(progress);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });
        mSeekBar_displayBrightness.setEnabled(false);

        // Display brightness mode
        mSwitch_brightnessMode = (Switch) findViewById(R.id.switch_brightnessMode);
        mSwitch_brightnessMode.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mDisplayManager.setBrightnessMode(DisplayManager.BRIGHTNESS_MODE_AUTOMATIC);
                } else {
                    mDisplayManager.setBrightnessMode(DisplayManager.BRIGHTNESS_MODE_MANUAL);
                }
            }
        });
        mSwitch_brightnessMode.setEnabled(false);

        // Display 2d3d mode
        mSwitch_displayMode = (Switch) findViewById(R.id.switch_displayMode);
        mSwitch_displayMode.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mDisplayManager.setDisplayMode(DisplayManager.DISPLAY_MODE_3D);
                } else {
                    mDisplayManager.setDisplayMode(DisplayManager.DISPLAY_MODE_2D);
                }
            }
        });
        mSwitch_displayMode.setEnabled(false);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }
}
