import * as tf from '@tensorflow/tfjs';
import RPSDataset from 'RockPaperScissors/RPSDataset';
import Webcam from 'RockPaperScissors/Webcam';

let model: tf.Sequential | undefined = undefined;

async function init() {
  document.body.innerHTML += `
  <div>
		<div>
			<video autoplay playsinline muted id="wc" width="224" height="224"></video>
		</div>
	</div>
	<button type="button" id="0">Rock</button>
	<button type="button" id="1">Paper</button>
	<button type="button" id="2">Scissors</button>
	<button type="button" id="3">Spock</button>
	<button type="button" id="4">Lizard</button>
	<div id="rocksamples">Rock Samples:</div>
	<div id="papersamples">Paper Samples:</div>
	<div id="scissorssamples">Scissors Samples:</div>
	<div id="spocksamples">Spock Samples:</div>
	<div id="lizardsamples">Lizard Samples:</div>
	<button type="button" id="train">Train Network</button>
	<div>Once training is complete, click 'Start Predicting' to see predictions, and 'Stop Predicting' to end</div>
	<button type="button" id="startPredicting">Start Predicting</button>
	<button type="button" id="stopPredicting">Stop Predicting</button>
  <button type="button" id="saveModel" >Download Model</button>
	<div id="prediction"></div>

  `;

  const webcam = new Webcam(document.getElementById('wc') as HTMLVideoElement);
  await webcam.setup();
  const dataset = new RPSDataset();
  const mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  const save = document.getElementById('saveModel') as HTMLButtonElement;
  save.onclick = saveModel;

  const rock = document.getElementById('0') as HTMLButtonElement;
  const paper = document.getElementById('1') as HTMLButtonElement;
  const scissors = document.getElementById('2') as HTMLButtonElement;
  const spock = document.getElementById('3') as HTMLButtonElement;
  const lizard = document.getElementById('4') as HTMLButtonElement;
  rock.onclick = () => handleButton('0');
  paper.onclick = () => handleButton('1');
  scissors.onclick = () => handleButton('2');
  spock.onclick = () => handleButton('3');
  lizard.onclick = () => handleButton('4');

  const trainButton = document.getElementById('train') as HTMLButtonElement;
  trainButton.onclick = doTraining;

  const startButton = document.getElementById('startPredicting') as HTMLButtonElement;
  startButton.onclick = startPredicting;
  const stopButton = document.getElementById('stopPredicting') as HTMLButtonElement;
  stopButton.onclick = stopPredicting;

  let lizardSamples = 0;
  let paperSamples = 0;
  let rockSamples = 0;
  let scissorsSamples = 0;
  let spockSamples = 0;
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
      case '3':
        spockSamples++;
        document.getElementById('spocksamples')!.innerText = 'Spock samples:' + spockSamples;
        break;
      case '4':
        lizardSamples++;
        document.getElementById('lizardsamples')!.innerText = 'Lizard samples:' + lizardSamples;
        break;
    }
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
    dataset.ys = null;
    dataset.encodeLabels(5);
    model = tf.sequential({
      layers: [
        tf.layers.flatten({inputShape: mobilenet.outputs[0]!.shape.slice(1)}),
        tf.layers.dense({units: 100, activation: 'relu'}),
        tf.layers.dense({units: 5, activation: 'softmax'}),
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
        case 3:
          predictionText = 'I see Spock';
          break;
        case 4:
          predictionText = 'I see Lizard';
          break;
      }
      document.getElementById('prediction')!.innerText = predictionText;

      predictedClass.dispose();
      await tf.nextFrame();
    }
  }
  predict.isPredicting = false;

  function saveModel() {
    model?.save('downloads://my_model');
  }
}

init();
