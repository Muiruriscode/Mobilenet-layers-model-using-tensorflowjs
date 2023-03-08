# Mobilenet classifier using mobilenet v2 layers model
This is a project using the pre-trained moblienet classifier from tensorflow hub. It is is already trained with to detect 1000 objects. You can add capabilities to the model to detect your own custom objects. This is  users a graph model enabling one to make changes to layers of mobilenet layers.

## About
The model detects two oblects based on which button has been clicked. If the first button is pressed it will gather data for an object and objects will be classified as class 1. If class two button is pressed the model wil detect objects which will be classified as class 2. 
More buttons bcan be added to classify more objects since the process is dynamic.

!["class 1"](./assets/class%201.png)

## Check video support and load video
```js
function hasGetUserMedia(){
    //check for mediaDevice support and getUser media support in the browser
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
}

function enableCam(){
    if(hasGetUserMedia()){
        const constraints = {
            video: true,
            width: 640,
            height: 480,
        }

    //allow streaming through the webcam using the browser
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream){
            VIDEO.srcObject = stream
            VIDEO.addEventListener('loadeddata', function(){
                videoPlaying = true
                ENABLE_CAM_BUTTON.classList.add('removed')
            })
        })
    }else{
        console.warn('getUserMedia() is not supported your browser')
    }
}

```
## Load layers model
```js
async function loadMobileNetFeatureModel(){
    const URL = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json';

    mobilenet = await tf.loadLayersModel(URL)
    STATUS.innerText = "MobileNet v2 loaded successfully"
    mobilenet.summary(null, null, customPrint)

  //create the base model from mobile net
    const layer = mobilenet.getLayer('global_average_pooling2d_1');
    mobileNetBase = tf.model({inputs: mobilenet.inputs, outputs: layer.output}); 
    mobileNetBase.summary();
  

    tf.tidy(function(){
    let answer = mobileNetBase.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
    })
}
```
## Gather data 
Gather data accoring to the button supported. Then change the state of gather data to eneble you to gather video data through the web cam.
```js
function gaterDataForClass() {
    let classNumber = parseInt(this.getAttribute('data-1hot'));
    gatherDataState = (gatherDataState === STOP_GATHER_DATA) ? classNumber : STOP_GATHER_DATA;
    dataGatherLoop();
  }

function calculateFeaturesOnCurrentFrame() {
    return tf.tidy(function() {
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
      let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor, 
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
      );
  
      let normalizedTensorFrame = resizedTensorFrame.div(255);
  
      return mobileNetBase.predict(normalizedTensorFrame.expandDims()).squeeze();
    });
  }

// 

function dataGatherLoop() {
    if (videoPlaying && gatherDataState !== STOP_GATHER_DATA) {
      let imageFeatures = calculateFeaturesOnCurrentFrame();
  
      trainingDataInputs.push(imageFeatures);
      trainingDataOutputs.push(gatherDataState);
      
      if (examplesCount[gatherDataState] === undefined) {
        examplesCount[gatherDataState] = 0;
      }
      examplesCount[gatherDataState]++;
  
      STATUS.innerText = '';
      for (let n = 0; n < CLASS_NAMES.length; n++) {
        STATUS.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
      }
  
      window.requestAnimationFrame(dataGatherLoop);
    }
  }
  

```

## Train and predict
This process involves training the model and then prdicting the value of objects detected in video frame. 
```js
async function trainAndPredict() {
    predict = false;
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    let inputsAsTensor = tf.stack(trainingDataInputs);
    
    let results = await model.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true,
      batchSize: 5,
      epochs: 5,
      callbacks: {onEpochEnd: logProgress}
    });
    
    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();

  //combine the two models
    let combinedModel = tf.sequential()
    combinedModel.add(mobileNetBase)
    combinedModel.add(model)

    combinedModel.compile({
      optimizer: 'adam',
      loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy'
    })
    combinedModel.summary()
    await combinedModel.save('downloads://my-model')
    
    predict = true;
    predictLoop();
  }
  
  // log progress to the console
  function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
  }
  
  function predictLoop() {
    if (predict) {
      tf.tidy(function() {
        let imageFeatures = calculateFeaturesOnCurrentFrame();
        let prediction = model.predict(imageFeatures.expandDims()).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();
        STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
      });
  
      window.requestAnimationFrame(predictLoop);
    }
  }
   
```
