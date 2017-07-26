# Tensorflow-Android
Create a docker image for build your own dataset on Android.

## Install docker
At first, we need to install docker on our PC and it's better to create your own docker account.    
See https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/ for installing docker-ce.

## Create a Docker container for building the TensorFlow Android demo

### Run a Docker container with all the TensorFlow dependencies

Use `docker run -it gcr.io/tensorflow/tensorflow:latest-devel` to download all the Tensorflow dependencies and you will see a **#** at your terminal.  

Check if python and tensorflow are working:  
```
python
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
apt-get update  
apt-get install wget
```

### Install Android Dependencies

NDK version recommended is 12b and API LEVEL >= 23  

```
# Download NDK and SDK
mkdir /android
cd /android  
wget https://dl.google.com/android/repository/tools_r25.2.3-linux.zip  
unzip tools_r25.2.3-linux.zip  
rm tools_r25.2.3-linux.zip  
wget https://dl.google.com/android/repository/android-ndk-r12b-linux-x86_64.zip  
unzip android-ndk-r12b-linux-x86_64.zip  
rm android-ndk-r12b-linux-x86_64.zip
# Install NDK and SDK
cd /android
tools/bin/sdkmanager "platforms;android-23"  
tools/bin/sdkmanager "build-tools;25.0.2"
```

### Configure TensorFlow build

```
cd /tensorflow
vim WORKSPACE
```    

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
cd /tensorflow
```

Use **--local_resources 4096,4.0,1.0 -j 1** to avoid out of memory errors.    

```
bazel build -c opt --local_resources 4096,4.0,1.0 -j 1 //tensorflow/examples/android:tensorflow_demo
```

Now that we’ve successfully completed our build, we can do our commit of our changes.

Open another ternimal and use `sudo docker ps` to see the CONTAINER ID.    
use `sudo docker commit CONTAINER ID <REPOSITORY>` to commit the changes.  

Use `sudo docker login` to log in your account and `sudo docker push <TAG>` to push the image to your Docker account.  
Use `exit` to exit the docker once you have saved your docker images

## Create your own image classifier

### Collect your data set

create a ~/tf_files/images folder and place each set of **jpeg** images in subdirectories (such as ~/tf_files/images/flower, ~/tf_files/images/computer, and so on).   

### Retrain the model to learn from your data set

```
$ sudo docker run -it -v $HOME/images:/images <REPOSITORY>
cd /tensorflow`  
python tensorflow/examples/image_retraining/retrain.py \
  --bottleneck_dir=/tf_files/bottlenecks \
  --how_many_training_steps 4000 \
  --model_dir=/tf_files/inception \
  --output_graph=/tf_files/retrained_graph.pb \
  --output_labels=/tf_files/retrained_labels.txt \
  --image_dir /tf_files/images
```

These commands will make TensorFlow download the inception model and retrain it to detect images from ~/tf_files/images.

And they will generate two files: the model in a protobuf file **retrained_graph.pb** and a label list **retrained_labels.txt** of all the objects that it can recognize.

### Optimize the model

```
python tensorflow/python/tools/optimize_for_inference.py \
  --input=/tf_files/retrained_graph.pb \
  --output=/tf_files/optimized_graph.pb \
  --input_names="Mul" \
  --output_names="final_result"
```

This will generate a ~/tf_files/optimized_graph.pb file which is removed all nodes that aren’t needed, among other optimizations.  

### Import the model in application
Now we are going to modifie some files.  

```
cd /tensorflow
cp /images/optimized_graph.pb /tensorflow/tensorflow/examples/android/assets
cp /images/retrained_labels.txt /tensorflow/tensorflow/examples/android/assets
```

These command can copy our retrained model and our retrained labels file into the Android demo app build.

```
cd /tensorflow/examples/android  
vim AndroidManifest.xml
```

Comment all these lines, if your don't want to install TF_detect and TF_stylize on your mobile.  

```java
   <!--<activity android:name="org.tensorflow.demo.DetectorActivity"
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
        </activity>-->
```  

Comment all these lines, and we will build the app with bazel, it will not download the model that we are not going to use.  

```
# "@inception5h//:model_files",  
# "@mobile_multibox//:model_files",  
# "@stylize//:model_files",  
```

Now we can build the app.

```
cd /tensorflow
bazel build -c opt --local_resources 4096,4.0,1.0 -j 1 //tensorflow/examples/android:tensorflow_demo
```

After that, we need to copy it to our shared mount so we can access it from outside the Docker container.

```
cp /tensorflow/bazel-bin/tensorflow/examples/android/tensorflow_demo.apk /tf_files
```

Do the same thing to commit our changes.  

Use `adb install -r $HOME/tf_files/tensorflow_demo.apk` to install the Android demo app.

## Create your own detection model

Now we are going to create our own model applied in TF_detect by using Yolo.

### prepare your dataset

Our dataset should be structured according to the VOC dataset:
  1. images should be **jpeg** format.  
  2. images are recommended to be renamed in such format: 2017_000001.jpg.  

Create a folder named `VOC` with 3 subfolders named `Annotation`, `ImageSets`, `JPEGImages`,
we will also create 3 subfolders of `ImageSets`: `Layout`, `Main`, `Segmentation`.

Put all images in `JPEGImages`.

Mark all the target area by using labelImg ([here](https://github.com/tzutalin/labelImg)), save all .xml file in Annotation folder.  

Create train.txt in Main folder which contain all image names.

### Use Darkflow

#### Download dependencies

Python3, tensorflow, numpy, opencv3, Cython.
#### Download weight files.

The weight files can be download in [here](https://drive.google.com/drive/folder/0B1tW_VtY7onidEwyQ2FtQVplWEU),
which include `yolo-full` and `yolo-tiny` of v1.0, `tiny-yolo-v1.1` of v1.1, and `yolo`, `tiny-yolo-voc` of v2.  
Create a **bin** folder and place the weight file in this folder.

#### Install Darkflow

```
$ git clone http://github/thtrieu/darkflow  
$ cd darkflow  
$ pip install .  
```

#### Training on your own dataset

1. Create a copy of the configuration file `tiny-yolo-voc.cfg` and rename it according to your preference like `tiny-yolo-voc-new.cfg`.
(It's crucial that you leave `tiny-yolo-voc.cfg` file unchanged)  
2. In `tiny-yolo-voc-new.cfg`, change classes in the [region] layer(the last layer) to the number of classes you are going to train for.    

    ```
    ...

    [region]
    anchors = 1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52
    bias_match=1
    classes=<your number of classes>
    coords=4
    num=5
    softmax=1

    ...
    ```

3. In `tiny-yolo-voc-new.cfg`, change filters in the [convolutional] layer(the second to last layer) to num*(classes + 5).  

    ```
    ...
    
    [convolutional]
    size=1
    stride=1
    pad=1
    filters=<num*(classes + 5)>
    activation=linear
    [region]
    anchors = 1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52
    
    ...
    ```

4. Change `labels.txt` to include the labels(s) you want to train on
(number of labels should be the same as the number of classes you set in `tiny-yolo-voc-new.cfg` file).
5. Reference the `tiny-yolo-voc-new.cfg` model when you train.

    `$ flow --model cfg/tiny-yolo-voc-new.cfg --load bin/tiny-yolo-voc.weights --train --annotation VOC/Annotation --dataset VOC/JPEGImages`

  You can use --lr for changing learning rate or use --epoch for changing number of epochs.

#### Prediction

Put some images which you are going to test for in `sample_img/`

```bash
# Forward all images in sample_img/ using the lastest checkpoing
$ flow --imgdir sample_img/ --model cfg/tiny-yolo-voc-new.cfg --load -1  
```

#### Save the built graph to a protobuf file(`.pb`)

During training, the script will occasionally save intermediate results into Tensorflow checkpoints, store in `ckpt/`.
To resume to any checkpoint before preforming training/testing, use `--load [checkpoint_num]` option,
if `checkpoint_num < 0`, darkflow will load the most recent save by parsing ckpt/checkpoint.

```bash
# Saving the lastest checkpoint to protobuf file
$ flow --model cfg/tiny-yolo-voc-new.cfg --load -1 --savepb
```
### Use the pre-trained model in Android

```bash
# run docker
$ sudo docker run -it -v $HOME/<FOLDER_CONTAIN_MODEL>:/<FOLDER_CONTAIN_MODEL> <REPOSITORY>
cp /<FOLDER_CONTAIN_MODEL>/tiny-yolo-voc-new.pb /tensorflow/tensorflow/examples/android/assets
```

```
cd /tensorflow/tensorflow/examples/android
vim AndroidManifest.xml
```

Uncomment all these lines if you have commented them.  

```java
   <activity android:name="org.tensorflow.demo.DetectorActivity"
                  android:screenOrientation="portrait"
                  android:label="@string/activity_name_detection">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
```

```
cd /tensorflow/tensorflow/examples/android/src/org/tensorflow/demo
vim DetectorActivity.java
```

Modify these 2 lines in DetectorActivity.java

```java
...

// change the path of yolo model file
private static final String YOLO_MODEL_FILE = "file:///android_asset/tiny-yolo-voc-new.pb";

...

// change USE_YOLO to true
private static final boolean USE_YOLO = true;

...
```

Modify these lines in TensorFlowYoloDetector.java

```java
...

// your number of classes
private static final NUM_CLASSES = <YOUR NUMBER OF CLASSES>;

private static final String[] LABELS = {
  // your labels
  label1,
  label2,
  label3
}

...
```

Now we can build the app.

```
cd /tensorflow
bazel build -c opt --local_resources 4096,4.0,1.0 -j 1 //tensorflow/examples/android:tensorflow_demo
```

After that, we need to copy it to our shared mount so we can access it from outside the Docker container.

```
cp /tensorflow/bazel-bin/tensorflow/examples/android/tensorflow_demo.apk /tf_files
```

Do the same thing to commit our changes.  

Use `adb install -r $HOME/tf_files/tensorflow_demo.apk` to install the Android demo app.
