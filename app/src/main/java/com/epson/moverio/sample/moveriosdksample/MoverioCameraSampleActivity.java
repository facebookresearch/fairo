package com.epson.moverio.sample.moveriosdksample;

import android.content.Context;
import android.graphics.Color;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.epson.moverio.hardware.camera.CameraDevice;
import com.epson.moverio.hardware.camera.CameraManager;
import com.epson.moverio.hardware.camera.CameraProperty;
import com.epson.moverio.hardware.camera.CaptureDataCallback;
import com.epson.moverio.hardware.camera.CaptureStateCallback;
import com.epson.moverio.hardware.sensor.SensorData;
import com.epson.moverio.hardware.sensor.SensorDataListener;
import com.epson.moverio.hardware.sensor.SensorManager;
import com.epson.moverio.util.PermissionHelper;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;
import fi.iki.elonen.NanoHTTPD;

class WebServer extends NanoHTTPD {

	private Bitmap img = null;
	private float[] acc = null;
	private float[] mag = null;
	private float[] gyro = null;
	private float[] light = null;
	private float[] la = null;
	private float[] grav = null;
	private float[] rv = null;
	public WebServer() {
		super(8080);
	}

	public void setImage(Bitmap img_) {
		img = img_;
	}
	public void setAcc(float[] acc_) {acc = acc_;}
	public void setMag(float[] mag_) {mag = mag_;}
	public void setGyro(float[] gyro_) {gyro = gyro_;}
	public void setLight(float[] light_) {light = light_;}
	public void setLa(float[] la_) {la = la_;}
	public void setGrav(float[] grav_) {grav = grav_;}
	public void setRv(float[] rv_) {rv = rv_;}

	@Override
	public Response serve(IHTTPSession session) {
		String uri = session.getUri();
		Log.w("Httpd", "|" + uri + "|");
		if (uri.equals("/sensors")) {
			String answer="";
			if (acc != null) {
				float[] data = acc;
				answer += "accelerometer," + String.format("%.4f", data[0]) + "," + String.format("%.4f", data[1]) + "," + String.format("%.4f", data[2]) + "\n";
			}
			if (mag != null) {
				float[] data = mag;
				answer += "magnetic_field," + String.format("%.4f", data[0]) + "," + String.format("%.4f", data[1]) + "," + String.format("%.4f", data[2]) + "\n";
			}
			if (gyro != null) {
				float[] data = gyro;
				answer += "gyro," + String.format("%.4f", data[0]) + "," + String.format("%.4f", data[1]) + "," + String.format("%.4f", data[2]) + "\n";
			}
			if (light != null) {
				float[] data = light;
				answer += "light," + String.format("%.4f", data[0]) + "\n";
			}
			if (la != null) {
				float[] data = la;
				answer += "linear_acceleration," + String.format("%.4f", data[0]) + "," + String.format("%.4f", data[1]) + "," + String.format("%.4f", data[2]) + "\n";
			}
			if (grav != null) {
				float[] data = grav;
				answer += "gravity," + String.format("%.4f", data[0]) + "," + String.format("%.4f", data[1]) + "," + String.format("%.4f", data[2]) + "\n";
			}
			if (rv != null) {
				float[] data = rv;
				answer += "rotation_vector," + String.format("%.4f", data[0]) + "," + String.format("%.4f", data[1]) + "," + String.format("%.4f", data[2]) + "\n";
			}
			return newFixedLengthResponse(answer);
		}

		if (uri.equals("/camera")) {
			if (img != null) {
				try {
					ByteArrayOutputStream stream = new ByteArrayOutputStream();
					if (img.compress(Bitmap.CompressFormat.JPEG, 50, stream)) {
						long cb = stream.size();
						InputStream is = new ByteArrayInputStream(stream.toByteArray());
						return newFixedLengthResponse(Response.Status.OK, "image/jpeg", is, is.available());
					}
				} catch (IOException exp) {
					String answer = "error!";
					return newFixedLengthResponse(answer);
				}

			}
		}
		String answer = "hi!";
		return newFixedLengthResponse(answer);
	}

}


public class MoverioCameraSampleActivity extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback {
	private final String TAG = this.getClass().getSimpleName();

	private SurfaceView mSurfaceView = null;
	private WebServer server;

	private Context mContext = null;

	private SensorManager mSensorManager = null;
	private SensorDataListener mSensorDataListener_acc = null;
	private SensorDataListener mSensorDataListener_mag = null;
	private SensorDataListener mSensorDataListener_gyro = null;
	private SensorDataListener mSensorDataListener_light = null;
	private SensorDataListener mSensorDataListener_la = null;
	private SensorDataListener mSensorDataListener_grav = null;
	private SensorDataListener mSensorDataListener_rv = null;


	private CameraManager mCameraManager = null;
	private CameraDevice mCameraDevice = null;
	private PermissionHelper mPermissionHelper = null;

	private ToggleButton mToggleButton_open = null;
	private ToggleButton mToggleButton_start = null;
	private ToggleButton mToggleButton_start_preview = null;
	private Switch mSwitch_autoExposure = null;
	private SeekBar mSeekBar_exposure = null;
	private SeekBar mSeekBar_brighness = null;
	private SeekBar mSeekBar_sharpness = null;
	private RadioGroup mRadioGroup_whitebalance = null;
	private RadioGroup mRadioGroup_powerLineFrequency = null;
	private Button mButton_takePicture = null;
	private ToggleButton mToggleButton_redording = null;
	private TextView mTextView_captureRate = null;
	private TextView mTextView_propertyInfo = null;
	private Button mButton_getProperty = null;
	private TextView mTextView_camApiResult = null;

	private int mCaptureCount = 0;
	private long mCaptureCountStart = 0;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		setContentView(R.layout.activity_moverio_camera_sample);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		mPermissionHelper = new PermissionHelper(this);
		mCameraManager = new CameraManager(this);
		mContext = this;
		mSensorManager = new SensorManager(mContext);


		server = new WebServer();
		try {
			server.start();
		} catch(IOException ioe) {
			Log.w("Httpd", "The server could not start.");
		}
		Log.w("Httpd", "Web server initialized.");
	}

	private CaptureStateCallback mCaptureStateCallback = null;
	private CaptureDataCallback mCaptureDataCallback = null;

	private TimerTask mTimerTask = new TimerTask(){
		@Override
		public void run() {
			mHandler.post( new Runnable() {
				public void run() {
					if(null != mTextView_captureRate) mTextView_captureRate.setText(String.valueOf(mFps) + "[fps]");
					else ;
				}
			});
		}
	};
	private Timer mTimer   = null;
	private Handler mHandler = new Handler();
	private float mFps = 0;

	private final Handler handler = new Handler();

	@Override
	protected void onStart() {
		super.onStart();

		mSensorDataListener_acc = new SensorDataListener() {
			@Override
			public void onSensorDataChanged(final SensorData data) {
				server.setAcc(data.values); //0-2
			}
		};
		try {
			mSensorManager.open(SensorManager.TYPE_ACCELEROMETER, mSensorDataListener_acc);
		} catch (IOException e) {
			e.printStackTrace();
		}

		mSensorDataListener_mag = new SensorDataListener() {
			@Override
			public void onSensorDataChanged(final SensorData data) {
				server.setMag(data.values); //0-2
			}
		};
		try{
			mSensorManager.open(SensorManager.TYPE_MAGNETIC_FIELD, mSensorDataListener_mag);
		} catch (IOException e) {
			e.printStackTrace();
		}

		mSensorDataListener_gyro = new SensorDataListener() {
			@Override
			public void onSensorDataChanged(final SensorData data) {
				server.setGyro(data.values); //0-2
			}
		};
		try {
			mSensorManager.open(SensorManager.TYPE_GYROSCOPE, mSensorDataListener_gyro);
		} catch (IOException e) {
			e.printStackTrace();
		}
		mSensorDataListener_light = new SensorDataListener() {
			@Override
			public void onSensorDataChanged(final SensorData data) {
				server.setLight(data.values); //0
			}
		};
		try {
			mSensorManager.open(SensorManager.TYPE_LIGHT, mSensorDataListener_light);
		} catch (IOException e) {
			e.printStackTrace();
		}
		mSensorDataListener_la = new SensorDataListener() {
			@Override
			public void onSensorDataChanged(final SensorData data) {
				server.setLa(data.values); //0-2
			}
		};
		try {
			mSensorManager.open(SensorManager.TYPE_LINEAR_ACCELERATION, mSensorDataListener_la);
		} catch (IOException e) {
			e.printStackTrace();
		}
		mSensorDataListener_grav = new SensorDataListener() {
			@Override
			public void onSensorDataChanged(final SensorData data) {
				server.setGrav(data.values); //0-2
			}
		};
		try{
			mSensorManager.open(SensorManager.TYPE_GRAVITY, mSensorDataListener_grav);
		} catch (IOException e) {
			e.printStackTrace();
		}
		mSensorDataListener_rv = new SensorDataListener() {
			@Override
			public void onSensorDataChanged(final SensorData data) {
				server.setRv(data.values); //0-2
			}
		};
		try {
			mSensorManager.open(SensorManager.TYPE_ROTATION_VECTOR, mSensorDataListener_rv);
		} catch (IOException e) {
			e.printStackTrace();
		}

		mSurfaceView = (SurfaceView) findViewById(R.id.surfaceView_preview);
		mTextView_camApiResult = (TextView) findViewById(R.id.textView_camApiResult);

		mTimer = new Timer(true);
		mTimer.schedule( mTimerTask, 1000, 1000);

		mTextView_captureRate = (TextView) findViewById(R.id.textView_rate);
		mTextView_captureRate.setTextColor(Color.WHITE);
		mToggleButton_open = (ToggleButton) findViewById(R.id.toggleButton_cameraOpenClose);
		mToggleButton_open.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
			@Override
			public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
				if(isChecked){
					try {
						mCaptureStateCallback = new CaptureStateCallback() {
							@Override
							public void onCaptureStarted() {
								Log.d(TAG, "onCaptureStarted");
								mCaptureCountStart = System.nanoTime();
								handler.post(new Runnable() {
									public void run() {
										mTextView_camApiResult.setText("onCaptureStarted");
									}
								});
							}

							@Override
							public void onCaptureStopped() {
								Log.d(TAG, "onCaptureStopped");
								handler.post(new Runnable() {
									public void run() {
										mTextView_camApiResult.setText("onCaptureStopped");
									}
								});
							}

							@Override
							public void onPreviewStarted() {
								Log.d(TAG, "onPreviewStarted");
								setInitView();
								handler.post(new Runnable() {
									public void run() {
										mTextView_camApiResult.setText("onPreviewStarted");
									}
								});
							}

							@Override
							public void onPreviewStopped() {
								Log.d(TAG, "onPreviewStopped");
								handler.post(new Runnable() {
									public void run() {
										mTextView_camApiResult.setText("onPreviewStopped");
									}
								});
							}

							@Override
							public void onRecordStarted() {
								Log.d(TAG, "onRecordStarted");
								handler.post(new Runnable() {
									public void run() {
										mTextView_camApiResult.setText("onRecordStarted");
									}
								});
							}

							@Override
							public void onRecordStopped() {
								Log.d(TAG, "onRecordStopped");
								handler.post(new Runnable() {
									public void run() {
										mTextView_camApiResult.setText("onRecordStopped");
									}
								});
							}

							@Override
							public void onPictureCompleted() {
								Log.d(TAG, "onPictureCompleted");
								handler.post(new Runnable() {
									public void run() {
										mTextView_camApiResult.setText("onPictureCompleted");
									}
								});
							}
						};
						mCaptureDataCallback = new CaptureDataCallback() {
							@Override
							public void onCaptureData(long timestamp, byte[] data) {

//								Log.d(TAG, "onCaptureData():"+timestamp+",data length="+data.length);
								mCaptureCount++;
								if((timestamp - mCaptureCountStart)/1000000000L >= 1) {
									mFps = (float)mCaptureCount/((timestamp - mCaptureCountStart)/1000000000L);
									mCaptureCount = 0;
									mCaptureCountStart = timestamp;
								}
								else ;

								Bitmap bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.RGB_565);
								ByteBuffer buffer = ByteBuffer.wrap(data);
								bitmap.copyPixelsFromBuffer(buffer);
								server.setImage(bitmap);
							}
						};

						mCameraDevice = mCameraManager.open(mCaptureStateCallback, mCaptureDataCallback, mSurfaceView.getHolder());
						if(null != mCameraDevice) {
							mTextView_camApiResult.setText("open");
							int width = 640;
							int height = 480;
							int fps = 15;
							CameraProperty property = mCameraDevice.getProperty();
							property.setCaptureSize(width, height);
							property.setCaptureFps(fps);
							int ret = mCameraDevice.setProperty(property);
							mTextView_camApiResult.setText("setProperty : "+ret);
						}
						else ;
					} catch (IOException e) {
						e.printStackTrace();
						mTextView_camApiResult.setText("IOException");
					}
				}
				else {
					mCameraManager.close(mCameraDevice);
					mTextView_camApiResult.setText("close");
				}
			}
		});

		mToggleButton_start = (ToggleButton) findViewById(R.id.toggleButton_cameraStartStop);
		mToggleButton_start.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
			@Override
			public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
				if(null == mCameraDevice) return ;
				if(isChecked){
					int ret = mCameraDevice.startCapture();
					mTextView_camApiResult.setText("startCapture : "+ret);
				}
				else {
					mCameraDevice.stopCapture();
					mTextView_camApiResult.setText("stopCapture");
				}
			}
		});

		mToggleButton_start_preview = (ToggleButton) findViewById(R.id.toggleButton_previewStartStop);
		mToggleButton_start_preview.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
			@Override
			public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
				if(null == mCameraDevice) return ;
				if(isChecked){
					int ret = mCameraDevice.startPreview();
					mTextView_camApiResult.setText("startPreview : "+ret);
				}
				else {
					mCameraDevice.stopPreview();
					mTextView_camApiResult.setText("stopPreview");
				}
			}
		});

		mSwitch_autoExposure = (Switch) findViewById(R.id.switch_autoExposure);
		mSwitch_autoExposure.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
			@Override
			public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
				if(null == mCameraDevice) return ;
				CameraProperty property = mCameraDevice.getProperty();
				if(isChecked){
					property.setExposureMode(CameraProperty.EXPOSURE_MODE_AUTO);
					mSeekBar_exposure.setEnabled(false);
				}
				else{
					property.setExposureMode(CameraProperty.EXPOSURE_MODE_MANUAL);
					mSeekBar_exposure.setEnabled(true);
				}
				int ret = mCameraDevice.setProperty(property);
				mTextView_camApiResult.setText("setProperty : "+ret);
			}
		});
		mSeekBar_exposure = (SeekBar) findViewById(R.id.seekBar_exposure);
		mSeekBar_exposure.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				if(null == mCameraDevice) return ;
				CameraProperty property = mCameraDevice.getProperty();
				property.setExposureStep(progress + property.getExposureStepMin());
				int ret = mCameraDevice.setProperty(property);
				mTextView_camApiResult.setText("setProperty : "+ret);
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {

			}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {

			}
		});

		mSeekBar_brighness = (SeekBar) findViewById(R.id.seekBar_brightness);
		mSeekBar_brighness.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				if(null == mCameraDevice) return ;
				CameraProperty property = mCameraDevice.getProperty();
				property.setBrightness(progress + property.getBrightnessMin());
				int ret = mCameraDevice.setProperty(property);
				mTextView_camApiResult.setText("setProperty : "+ret);
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {

			}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {

			}
		});

		mSeekBar_sharpness = (SeekBar) findViewById(R.id.seekBar_sharpness);
		mSeekBar_sharpness.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				if(null == mCameraDevice) return ;
				CameraProperty property = mCameraDevice.getProperty();
				property.setSharpness(progress + property.getSharpnessMin());
				int ret = mCameraDevice.setProperty(property);
				mTextView_camApiResult.setText("setProperty : "+ret);
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {

			}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {

			}
		});

		mRadioGroup_whitebalance = (RadioGroup) findViewById(R.id.radioGroup_wb);
		mRadioGroup_whitebalance.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
			@Override
			public void onCheckedChanged(RadioGroup group, int checkedId) {
				if(null == mCameraDevice) return ;
				CameraProperty property = mCameraDevice.getProperty();
				switch (checkedId){
					case R.id.radioButton_auto:
						property.setWhiteBalanceMode(CameraProperty.WHITE_BALANCE_MODE_AUTO);
						break;
					case R.id.radioButton_cloudyDaylight:
						property.setWhiteBalanceMode(CameraProperty.WHITE_BALANCE_MODE_CLOUDY_DAYLIGHT);
						break;
					case R.id.radioButton_daylight:
						property.setWhiteBalanceMode(CameraProperty.WHITE_BALANCE_MODE_DAYLIGHT);
						break;
					case R.id.radioButton_fluorescent:
						property.setWhiteBalanceMode(CameraProperty.WHITE_BALANCE_MODE_FLUORESCENT);
						break;
					case R.id.radioButton_incandescent:
						property.setWhiteBalanceMode(CameraProperty.WHITE_BALANCE_MODE_INCANDESCENT);
						break;
					case R.id.radioButton_twilight:
						property.setWhiteBalanceMode(CameraProperty.WHITE_BALANCE_MODE_TWILIGHT);
						break;
					default:
						Log.w(TAG, "id="+checkedId);
						break;
				}
				int ret = mCameraDevice.setProperty(property);
				mTextView_camApiResult.setText("setProperty : "+ret);
			}
		});
		mRadioGroup_whitebalance.clearCheck();

		mRadioGroup_powerLineFrequency = (RadioGroup) findViewById(R.id.radioGroup_powerLineFrequency);
		mRadioGroup_powerLineFrequency.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
			@Override
			public void onCheckedChanged(RadioGroup group, int checkedId) {
				if(null == mCameraDevice) return ;
				CameraProperty property = mCameraDevice.getProperty();
				switch (checkedId){
					case R.id.radioButton_50Hz:
						property.setPowerLineFrequencyControlMode(CameraProperty.POWER_LINE_FREQUENCY_CONTROL_MODE_50HZ);
						break;
					case R.id.radioButton_60Hz:
						property.setPowerLineFrequencyControlMode(CameraProperty.POWER_LINE_FREQUENCY_CONTROL_MODE_60HZ);
						break;
					default:
						Log.w(TAG, "id="+checkedId);
						break;
				}
				int ret = mCameraDevice.setProperty(property);
				mTextView_camApiResult.setText("setProperty : "+ret);
			}
		});
		mRadioGroup_powerLineFrequency.clearCheck();

		mButton_takePicture = (Button) findViewById(R.id.button_takePicture);
		mButton_takePicture.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				if(null == mCameraDevice) return ;
				String fileName = "image_" + new SimpleDateFormat("yyyyMMddHHmmss").format(new Date(System.currentTimeMillis())) + ".jpg";
				int ret = mCameraDevice.takePicture(new File(Environment.getExternalStorageDirectory().getAbsolutePath(), fileName));
				mTextView_camApiResult.setText("takePicture : "+ret + Environment.getExternalStorageDirectory().getAbsolutePath() + fileName);
			}
		});

		mToggleButton_redording = (ToggleButton) findViewById(R.id.toggleButton_redording);
		mToggleButton_redording.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
			@Override
			public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
				if(null == mCameraDevice) return ;
				if(isChecked){
					String fileName = "movie_" + new SimpleDateFormat("yyyyMMddHHmmss").format(new Date(System.currentTimeMillis())) + ".mp4";
					int ret = mCameraDevice.startRecord(new File(Environment.getExternalStorageDirectory().getAbsolutePath(), fileName));
					mTextView_camApiResult.setText("startRecord : "+ret);
				}
				else {
					mCameraDevice.stopRecord();
					mTextView_camApiResult.setText("stopRecord");
				}
			}
		});

		mTextView_propertyInfo = (TextView) findViewById(R.id.textView_propertyInfo);
		mButton_getProperty = (Button) findViewById(R.id.button_getProperty);
		mButton_getProperty.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				if(null != mCameraDevice){
					CameraProperty property = mCameraDevice.getProperty();
					if(null != property){
						String str = "";
						str += "Size         : " + property.getCaptureSize()[0] + "x" + property.getCaptureSize()[1] + " / " + property.getCaptureFps() + " [fps]\n";
						str += "Exposure     : " + (property.getExposureMode().equals(CameraProperty.EXPOSURE_MODE_AUTO) ? property.getExposureMode() + "(" + property.getExposureStep() + ")" : property.getExposureStep()) + "\n";
						str += "Brightness   : " + property.getBrightness() + "\n";
						str += "Sharpness    : " + property.getSharpness() + "\n";
						str += "Whitebalance : " + (property.getWhiteBalanceMode().equals(CameraProperty.WHITE_BALANCE_MODE_AUTO) ? property.getWhiteBalanceMode() + "(" + property.getWhiteBalanceTemperature() + ")"  : property.getWhiteBalanceTemperature()) + "\n";
						str += "PowerLineFrequency : " + property.getPowerLineFrequencyControlMode() + "\n";
						str += "CaptureDataFormat ： " + property.getCaptureDataFormat() + "\n";
						mTextView_propertyInfo.setText(str);
					}
					else {
						mTextView_propertyInfo.setText("null");
					}
				}
			}
		});
	}

	private void setInitView(){
		if(null == mCameraDevice) return ;
		// Exposure
		mSeekBar_exposure.setMax(mCameraDevice.getProperty().getExposureStepMax() - mCameraDevice.getProperty().getExposureStepMin());

		if(mCameraDevice.getProperty().getExposureMode().equals(CameraProperty.EXPOSURE_MODE_AUTO)) {
			mSwitch_autoExposure.setChecked(true);
			mSeekBar_exposure.setEnabled(false);
		}
		else {
			mSwitch_autoExposure.setChecked(false);
			mSeekBar_exposure.setEnabled(true);
			mSeekBar_exposure.setProgress(mCameraDevice.getProperty().getExposureStep() - mCameraDevice.getProperty().getExposureStepMin());
		}
		// Brightness
		mSeekBar_brighness.setMax(mCameraDevice.getProperty().getBrightnessMax() - mCameraDevice.getProperty().getBrightnessMin());
		mSeekBar_brighness.setProgress(mCameraDevice.getProperty().getBrightness() - mCameraDevice.getProperty().getBrightnessMin());
		// Sharpness
		mSeekBar_sharpness.setMax(mCameraDevice.getProperty().getSharpnessMax() - mCameraDevice.getProperty().getSharpnessMin());
		mSeekBar_sharpness.setProgress(mCameraDevice.getProperty().getSharpness() - mCameraDevice.getProperty().getSharpnessMin());
		// White balance
		if(mCameraDevice.getProperty().getWhiteBalanceMode().equals(CameraProperty.WHITE_BALANCE_MODE_AUTO)) {
		}
		else {
		}
		// Power line frequency
		if(mCameraDevice.getProperty().getPowerLineFrequencyControlMode().equals(CameraProperty.POWER_LINE_FREQUENCY_CONTROL_MODE_50HZ)){
			mRadioGroup_powerLineFrequency.check(R.id.radioButton_50Hz);
		}
		else if(mCameraDevice.getProperty().getPowerLineFrequencyControlMode().equals(CameraProperty.POWER_LINE_FREQUENCY_CONTROL_MODE_60HZ)){
			mRadioGroup_powerLineFrequency.check(R.id.radioButton_60Hz);
		}
		else {
			Log.e(TAG, "Unknown power frequency mode.....");
		}
	}

	@Override
	protected void onRestart()	{
		super.onRestart();

		if(null != mCameraDevice) {
			mCameraDevice.stopCapture();
		} else;
	}

	@Override
	protected void onStop(){
		super.onStop();

		if(null != mCameraDevice) {
			mCameraDevice.stopCapture();
		} else ;
	}

	@Override
	protected void onDestroy()	{
		super.onDestroy();

		if(null != mCameraDevice) {
			mCameraManager.close(mCameraDevice);
		} else;

		if (server != null)
			server.stop();

		mTimer.cancel();
		mTimer = null;
	}

	@Override
	public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
		super.onRequestPermissionsResult(requestCode, permissions, grantResults);
		mPermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		getMenuInflater().inflate(R.menu.option, menu);
		return super.onCreateOptionsMenu(menu);
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		int width = 640;
		int height = 480;
		int fps = 15;
		if(null != mCameraDevice) {
			CameraProperty property = mCameraDevice.getProperty();
			property.setCaptureSize(width, height);
			property.setCaptureFps(fps);
			int ret = mCameraDevice.setProperty(property);
			mTextView_camApiResult.setText("setProperty : "+ret);
		}
		else ;

		return super.onOptionsItemSelected(item);
	}
}
