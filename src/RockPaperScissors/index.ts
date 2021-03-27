import * as tf from '@tensorflow/tfjs';

import RPSDataset from './RPSDataset';
import Webcam from './Webcam';

let model: tf.Sequential | undefined = undefined;

async function init() {
  document.body.innerHTML += `
  <div>
		<div>
			<video autoplay playsinline muted id="wc" width="224" height="224"></video>
		</div>
	</div>
	<button type="button" id="0" >Rock</button>
	<button type="button" id="1" >Paper</button>
	<button type="button" id="2" >Scissors</button>
	<div id="rocksamples">Rock Samples:</div>
	<div id="papersamples">Paper Samples:</div>
	<div id="scissorssamples">Scissors Samples:</div>
	<button type="button" id="train">Train Network</button>
	<div id="dummy">Once training is complete, click 'Start Predicting' to see predictions, and 'Stop Predicting' to end</div>
	<button type="button" id="startPredicting">Start Predicting</button>
	<button type="button" id="stopPredicting">Stop Predicting</button>
	<div id="prediction"></div>
  `;

  const webcam = new Webcam(document.getElementById('wc') as HTMLVideoElement);
  await webcam.setup();
  const dataset = new RPSDataset();
  const mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  const rock = document.getElementById('0') as HTMLButtonElement;
  const paper = document.getElementById('1') as HTMLButtonElement;
  const scissors = document.getElementById('2') as HTMLButtonElement;
  rock.onclick = () => handleButton('0');
  paper.onclick = () => handleButton('1');
  scissors.onclick = () => handleButton('2');

  const trainButton = document.getElementById('train') as HTMLButtonElement;
  trainButton.onclick = doTraining;

  const startButton = document.getElementById('startPredicting') as HTMLButtonElement;
  startButton.onclick = startPredicting;
  const stopButton = document.getElementById('stopPredicting') as HTMLButtonElement;
  stopButton.onclick = stopPredicting;

  let rockSamples = 0;
  let paperSamples = 0;
  let scissorsSamples = 0;
  function handleButton(id: string) {
    switch (id) {
      case '0':
        rockSamples++;
        document.getElementById('rocksamples')!.innerText = 'Rock samples:' + rockSamples;
        break;
      case '1':
        paperSamples++;
        document.getElementById('papersamples')!.innerText = 'Paper samples:' + paperSamples;
        break;
      case '2':
        scissorsSamples++;
        document.getElementById('scissorssamples')!.innerText =
          'Scissors samples:' + scissorsSamples;
        break;
    }
    console.log(id, '??');
    const label = parseInt(id, 10);
    const img = webcam.capture();
    dataset.addExample(mobilenet.predict(img) as tf.Tensor2D, label);
  }

  async function loadMobilenet(): Promise<tf.LayersModel> {
    const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',
    );
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  }

  function doTraining() {
    console.log(mobilenet.outputs, mobilenet.outputs[0]);
    dataset.ys = null;
    dataset.encodeLabels(3);
    model = tf.sequential({
      layers: [
        tf.layers.flatten({inputShape: mobilenet.outputs[0]!.shape.slice(1)}),
        tf.layers.dense({units: 100, activation: 'relu'}),
        tf.layers.dense({units: 3, activation: 'softmax'}),
      ],
    });
    const optimizer = tf.train.adam(0.0001);
    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
    console.log('model.fit');
    model.fit(dataset.xs!, dataset.ys!, {
      epochs: 10,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          console.log('LOSS: ' + logs?.loss?.toFixed(5));
        },
      },
    });
  }

  function startPredicting() {
    predict.isPredicting = true;
    console.log('startPredicting');
    console.log(predict.isPredicting);
    predict();
  }

  function stopPredicting() {
    console.log('stopPredicting');
    predict.isPredicting = false;
    predict();
  }

  async function predict() {
    while (predict.isPredicting) {
      if (!model) {
        await tf.nextFrame();
        continue;
      }
      const predictedClass = tf.tidy(() => {
        const img = webcam.capture();
        const activation = mobilenet.predict(img);
        const predictions = model!.predict(activation) as tf.Tensor<tf.Rank>;
        return predictions.as1D().argMax();
      });
      const classId = (await predictedClass.data())[0];
      let predictionText = '';
      switch (classId) {
        case 0:
          predictionText = 'I see Rock';
          break;
        case 1:
          predictionText = 'I see Paper';
          break;
        case 2:
          predictionText = 'I see Scissors';
          break;
      }
      document.getElementById('prediction')!.innerText = predictionText;
      predictedClass.dispose();
      await tf.nextFrame();
    }
  }
  predict.isPredicting = false;
}

init();
