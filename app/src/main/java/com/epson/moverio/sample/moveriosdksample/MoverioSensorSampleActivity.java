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
import android.os.Handler;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.TextView;

import com.epson.moverio.hardware.sensor.SensorData;
import com.epson.moverio.hardware.sensor.SensorDataListener;
import com.epson.moverio.hardware.sensor.SensorManager;

import java.io.IOException;

public class MoverioSensorSampleActivity extends Activity {
    private final String TAG = this.getClass().getSimpleName();

    private Context mContext = null;

    private SensorManager mSensorManager = null;

    private CheckBox mCheckBox_acc = null;
    private CheckBox mCheckBox_mag = null;
    private CheckBox mCheckBox_gyro = null;
    private CheckBox mCheckBox_light = null;
    private CheckBox mCheckBox_la = null;
    private CheckBox mCheckBox_grav = null;
    private CheckBox mCheckBox_rv = null;
    private TextView mTextView_accResult = null;
    private TextView mTextView_magResult = null;
    private TextView mTextView_gyroResult = null;
    private TextView mTextView_lightResult = null;
    private TextView mTextView_laResult = null;
    private TextView mTextView_gravResult = null;
    private TextView mTextView_rvResult = null;
    private SensorDataListener mSensorDataListener_acc = null;
    private SensorDataListener mSensorDataListener_mag = null;
    private SensorDataListener mSensorDataListener_gyro = null;
    private SensorDataListener mSensorDataListener_light = null;
    private SensorDataListener mSensorDataListener_la = null;
    private SensorDataListener mSensorDataListener_grav = null;
    private SensorDataListener mSensorDataListener_rv = null;

    private final Handler handler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_moverio_sensor_sample);

        mContext = this;

        mSensorManager = new SensorManager(mContext);

        // Accelerometer sensor.
        mTextView_accResult = (TextView) findViewById(R.id.textView_accResult);
        mCheckBox_acc = (CheckBox) findViewById(R.id.checkBox_acc);
        mCheckBox_acc.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mSensorDataListener_acc = new SensorDataListener() {
                        @Override
                        public void onSensorDataChanged(final SensorData data) {
                            handler.post(new Runnable() {
                                public void run() {
                                    mTextView_accResult.setText(String.format("%.4f", data.values[0]) + "," + String.format("%.4f", data.values[1]) + "," + String.format("%.4f", data.values[2]));
                                }
                            });
                        }
                    };
                    try {
                        mSensorManager.open(SensorManager.TYPE_ACCELEROMETER, mSensorDataListener_acc);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    mSensorManager.close(mSensorDataListener_acc);
                    mSensorDataListener_acc = null;
                }
            }
        });

        // Magnetic field sensor.
        mTextView_magResult = (TextView) findViewById(R.id.textView_magResult);
        mCheckBox_mag = (CheckBox) findViewById(R.id.checkBox_mag);
        mCheckBox_mag.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mSensorDataListener_mag = new SensorDataListener() {
                        @Override
                        public void onSensorDataChanged(final SensorData data) {
                            handler.post(new Runnable() {
                                public void run() {
                                    mTextView_magResult.setText(String.format("%.4f", data.values[0]) + "," + String.format("%.4f", data.values[1]) + "," + String.format("%.4f", data.values[2]));
                                }
                            });
                        }
                    };
                    try{
                        mSensorManager.open(SensorManager.TYPE_MAGNETIC_FIELD, mSensorDataListener_mag);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    mSensorManager.close(mSensorDataListener_mag);
                }
            }
        });

        // Gyroscope sensor.
        mTextView_gyroResult = (TextView) findViewById(R.id.textView_gyroResult);
        mCheckBox_gyro = (CheckBox) findViewById(R.id.checkBox_gyro);
        mCheckBox_gyro.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mSensorDataListener_gyro = new SensorDataListener() {
                        @Override
                        public void onSensorDataChanged(final SensorData data) {
                            handler.post(new Runnable() {
                                public void run() {
                                    mTextView_gyroResult.setText(String.format("%.4f", data.values[0]) + "," + String.format("%.4f", data.values[1]) + "," + String.format("%.4f", data.values[2]));
                                }
                            });
                        }
                    };
                    try {
                        mSensorManager.open(SensorManager.TYPE_GYROSCOPE, mSensorDataListener_gyro);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    mSensorManager.close(mSensorDataListener_gyro);
                    mSensorDataListener_gyro = null;
                }
            }
        });

        // Light sensor.
        mTextView_lightResult = (TextView) findViewById(R.id.textView_lightResult);
        mCheckBox_light = (CheckBox) findViewById(R.id.checkBox_light);
        mCheckBox_light.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mSensorDataListener_light = new SensorDataListener() {
                        @Override
                        public void onSensorDataChanged(final SensorData data) {
                            handler.post(new Runnable() {
                                public void run() {
                                    mTextView_lightResult.setText(String.format("%.4f", data.values[0]));
                                }
                            });
                        }
                    };
                    try {
                        mSensorManager.open(SensorManager.TYPE_LIGHT, mSensorDataListener_light);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    mSensorManager.close(mSensorDataListener_light);
                    mSensorDataListener_light = null;
                }
            }
        });

        // Linear accelerometer sensor.
        mTextView_laResult = (TextView) findViewById(R.id.textView_laResult);
        mCheckBox_la = (CheckBox) findViewById(R.id.checkBox_la);
        mCheckBox_la.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mSensorDataListener_la = new SensorDataListener() {
                        @Override
                        public void onSensorDataChanged(final SensorData data) {
                            handler.post(new Runnable() {
                                public void run() {
                                    mTextView_laResult.setText(String.format("%.4f", data.values[0]) + "," + String.format("%.4f", data.values[1]) + "," + String.format("%.4f", data.values[2]));
                                }
                            });
                        }
                    };
                    try {
                        mSensorManager.open(SensorManager.TYPE_LINEAR_ACCELERATION, mSensorDataListener_la);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    mSensorManager.close(mSensorDataListener_la);
                    mSensorDataListener_la = null;
                }
            }
        });

        // Gravity sensor.
        mTextView_gravResult = (TextView) findViewById(R.id.textView_gravResult);
        mCheckBox_grav = (CheckBox) findViewById(R.id.checkBox_grav);
        mCheckBox_grav.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mSensorDataListener_grav = new SensorDataListener() {
                        @Override
                        public void onSensorDataChanged(final SensorData data) {
                            handler.post(new Runnable() {
                                public void run() {
                                    mTextView_gravResult.setText(String.format("%.4f", data.values[0]) + "," + String.format("%.4f", data.values[1]) + "," + String.format("%.4f", data.values[2]));
                                }
                            });
                        }
                    };
                    try{
                        mSensorManager.open(SensorManager.TYPE_GRAVITY, mSensorDataListener_grav);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    mSensorManager.close(mSensorDataListener_grav);
                    mSensorDataListener_grav = null;
                }
            }
        });

        // Rotation vector sensor.
        mTextView_rvResult = (TextView) findViewById(R.id.textView_rvResult);
        mCheckBox_rv = (CheckBox) findViewById(R.id.checkBox_rv);
        mCheckBox_rv.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mSensorDataListener_rv = new SensorDataListener() {
                        @Override
                        public void onSensorDataChanged(final SensorData data) {
                            handler.post(new Runnable() {
                                public void run() {
                                    mTextView_rvResult.setText(String.format("%.4f", data.values[0]) + "," + String.format("%.4f", data.values[1]) + "," + String.format("%.4f", data.values[2]) + "," + String.format("%.4f", data.values[3]));
                                }
                            });
                        }
                    };
                    try {
                        mSensorManager.open(SensorManager.TYPE_ROTATION_VECTOR, mSensorDataListener_rv);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    mSensorManager.close(mSensorDataListener_rv);
                    mSensorDataListener_rv = null;
                }
            }
        });
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }
}
