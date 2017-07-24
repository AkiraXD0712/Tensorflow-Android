# Tensorflow-Android
Create a docker image for build your own dataset on Android(TF_classify).   
## Installation
At first, we need to install docker on our PC and it's better to create your own docker account.    
See https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/ for installing docker-ce.  
## Create a Docker container for building the TensorFlow Android demo
### Run a Docker container with all the TensorFlow dependencies 
Use `docker run -it gcr.io/tensorflow/tensorflow:latest-devel` to download all the Tensorflow dependencies and you will see a **#** at your ternimal.  
  
Check python and tensorflow are working:  
```
# python
>> import tensorflow as tf  
>> hello = tf.constant('Hello world')  
>> sess = tf.Session()
>> print(sess.run(hello))
```
  
If tensorflow works, you will see 
>Hello World  
>
### Install wget:  
```
# apt-get update  
# apt-get install wget 
```
### Install Android Dependencies
NDK version recommanded is 12b and API LEVEL >= 23  
  
We are going to download NDK and SDK
```
# mkdir /android 
# cd /android  
# wget https://dl.google.com/android/repository/tools_r25.2.3-linux.zip  
# unzip tools_r25.2.3-linux.zip  
# rm tools_r25.2.3-linux.zip  
# wget https://dl.google.com/android/repository/android-ndk-r12b-linux-x86_64.zip  
# unzip android-ndk-r12b-linux-x86_64.zip  
# rm android-ndk-r12b-linux-x86_64.zip
```
And now we are going to install NDK and SDK  
```
# cd /android 
# tools/bin/sdkmanager "platforms;android-23"  
# tools/bin/sdkmanager "build-tools;25.0.2"
```  
### Configure TensorFlow build
```
# cd /tensorflow
# vi WORKSPACE
```  
(For downloading vi: use `# apt-get install vi`)  
  
Remove `#` to uncomment all these lines  
```
android_sdk_repository(  
  name = “androidsdk”,  
  api_level = 23,  
  build_tools_version = “25.0.2”,  
  path = “/android”,  
)  
android_ndk_repository(  
  name=”androidndk”,  
  path=”/android/android-ndk-r12b”,  
  api_level=21  
)  
```
### Building TensorFlow Android demo  
```
# cd /tensorflow
```
Use **--local_resources 4096,4.0,1.0 -j 1** to avoid out of memory errors.    
```
# bazel build -c opt --local_resources 4096,4.0,1.0 -j 1 //tensorflow/examples/android:tensorflow_demo
```
  
Now that we’ve successfully completed our build, we can do our commit of our changes.  
  
Open another ternimal and use `sudo docker ps` to see the CONTAINER ID.    
use `$ sudo docker commit CONTAINER ID <REPOSITORY>` to commit the changes.  

Use `$ sudo docker login` to log in your account and `$ sudo docker push <TAG>` to push the image to your Docker account.  
Use `# exit` to exit the docker once you have saved your docker images
## Create your own image classifier 
### Collect your data set  
create a ~/tf_files/images folder and place each set of **jpeg** images in subdirectories (such as ~/tf_files/images/flower, ~/tf_files/images/computer, and so on).   
### Retrain the model to learn from your data set
```
$ sudo docker run -it -v $HOME/images:/images <REPOSITORY> 
# cd /tensorflow`  
# python tensorflow/examples/image_retraining/retrain.py \
  --bottleneck_dir=/tf_files/bottlenecks \
  --how_many_training_steps 4000 \
  --model_dir=/tf_files/inception \
  --output_graph=/tf_files/retrained_graph.pb \
  --output_labels=/tf_files/retrained_labels.txt \
  --image_dir /tf_files/images
```
These commands will make TensorFlow download the inception model and retrain it to detect images from ~/tf_files/images. And they will generate two files: the model in a protobuf file **retrained_graph.pb** and a label list **retrained_labels.txt** of all the objects that it can recognize.  
### Optimize the model
```
# python tensorflow/python/tools/optimize_for_inference.py \
  --input=/tf_files/retrained_graph.pb \
  --output=/tf_files/optimized_graph.pb \
  --input_names="Mul" \
  --output_names="final_result"
```
This will generate a ~/tf_files/optimized_graph.pb file which is removed all nodes that aren’t needed, among other optimizations.   
### Import the model in application
Now we are going to modifie some files.  
```
# cd /tensorflow
# cp /images/optimized_graph.pb /tensorflow/tensorflow/examples/android/assets
# cp /images/retrained_labels.txt /tensorflow/tensorflow/examples/android/assets
``` 
These command can copy our retrained model and our retrained labels file into the Android demo app build.    
```
# cd /tensorflow/examples/android  
# vi AndroidManifest.xml
```
  
Comment all these lines, if your don't want to install TF_detect and TF_stylize on your mobile.  
```
   <!--     <activity android:name="org.tensorflow.demo.DetectorActivity"
                  android:screenOrientation="portrait"
                  android:label="@string/activity_name_detection">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

        <activity android:name="org.tensorflow.demo.StylizeActivity"
                  android:screenOrientation="portrait"
                  android:label="@string/activity_name_stylize">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>   -->
```  
  
Comment all these lines, and we will build the app with bazel, it will not download the model that we are not going to use.  
```
# "@inception5h//:model_files",  
# "@mobile_multibox//:model_files",  
# "@stylize//:model_files",  
```
  
Now we can build the app.  
```
# bazel build -c opt --local_resources 4096,4.0,1.0 -j 1 //tensorflow/examples/android:tensorflow_demo
```
  
After that, we need to copy it to our shared mount so we can access it from outside the Docker container.  
```
# cp /tensorflow/bazel-bin/tensorflow/examples/android/tensorflow_demo.apk /tf_files
```
  
Do the same thing to commit our changes.  
  
Use `adb install -r $HOME/tf_files/tensorflow_demo.apk` to install the Android demo app.  
##Create your own detection model
Now we are going to create our own model applied in TF_detect by using Yolo.
###pre



