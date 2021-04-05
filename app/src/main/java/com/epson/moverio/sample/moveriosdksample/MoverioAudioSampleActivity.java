package com.epson.moverio.sample.moveriosdksample;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.epson.moverio.hardware.audio.AudioManager;

import java.io.IOException;

/**
 * Created by EPSON on 2019/01/09.
 */
public class MoverioAudioSampleActivity extends Activity {
    private final String TAG = this.getClass().getSimpleName();

    private Context mContext = null;

    private AudioManager mAudioManager = null;

    private ToggleButton mToggleButton_audioOpenClose = null;
    private Button mButton_volumeDown = null;
    private Button mButton_volumeUp = null;
    private TextView mTextView_volume = null;
    private ToggleButton mToggleButton_volumeLimit = null;
//    private SeekBar mSeekBar_audioVolume = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_moverio_audio_sample);

        mContext = this;

        mAudioManager = new AudioManager(mContext);
        mToggleButton_audioOpenClose = (ToggleButton) findViewById(R.id.toggleButton_audioOpenClose);
        mToggleButton_audioOpenClose.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                mButton_volumeDown.setEnabled(isChecked);
                mButton_volumeUp.setEnabled(isChecked);
                mToggleButton_volumeLimit.setEnabled(isChecked);
                if (isChecked) {
                    try {
                        mAudioManager.open();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    mAudioManager.close();
                }
            }
        });

        // Audio volume control
        mTextView_volume = (TextView) findViewById(R.id.textView_volume);
        mButton_volumeDown = (Button) findViewById(R.id.button_volumeDown);
        mButton_volumeDown.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int prevVolume = 0, nowVolume = 0;
                int ret = mAudioManager.getVolume();
                if(ret < 0){
                    Log.e(TAG, "Fail to get prev volume.");
                    mTextView_volume.setText("err1");
                    return ;
                }
                else ;
                prevVolume = ret;
                ret = mAudioManager.setVolume(prevVolume - 1);
                if(ret < 0){
                    Log.e(TAG, "Fail to set volume.");
                    mTextView_volume.setText("err2");
                    return ;
                }
                else ;
                ret = mAudioManager.getVolume();
                if(ret < 0){
                    Log.e(TAG, "Fail to get now volume.");
                    mTextView_volume.setText("err3");
                    return ;
                }
                else ;
                nowVolume = ret;
                Log.d(TAG, "volume:now = " + nowVolume + ", prev = " + prevVolume + "  (ret=" + ret + ")");
                mTextView_volume.setText(String.valueOf(nowVolume));
            }
        });
        mButton_volumeUp = (Button) findViewById(R.id.button_volumeUp);
        mButton_volumeUp.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int prevVolume = 0, nowVolume = 0;
                int ret = mAudioManager.getVolume();
                if(ret < 0){
                    Log.e(TAG, "Fail to get prev volume.");
                    mTextView_volume.setText("err1");
                    return ;
                }
                else ;
                prevVolume = ret;
                ret = mAudioManager.setVolume(prevVolume + 1);
                if(ret < 0){
                    Log.e(TAG, "Fail to set volume.");
                    mTextView_volume.setText("err2");
                    return ;
                }
                else ;
                ret = mAudioManager.getVolume();
                if(ret < 0){
                    Log.e(TAG, "Fail to get now volume.");
                    mTextView_volume.setText("err3");
                    return ;
                }
                else ;
                nowVolume = ret;
                Log.d(TAG, "volume:now = " + nowVolume + ", prev = " + prevVolume + "  (ret=" + ret + ")");
                mTextView_volume.setText(String.valueOf(nowVolume));
            }
        });

        // Audio volume limit
        mToggleButton_volumeLimit = (ToggleButton) findViewById(R.id.toggleButton_volumeLimitOnOff);
        mToggleButton_volumeLimit.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mAudioManager.setVolumeLimitMode(AudioManager.VOLUME_LIMIT_MODE_ON);
                } else {
                    mAudioManager.setVolumeLimitMode(AudioManager.VOLUME_LIMIT_MODE_OFF);
                }
            }
        });
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }
}
