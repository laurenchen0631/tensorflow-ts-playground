import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {MnistData} from './data';

const model = getModel();

function createElements() {
  document.body.innerHTML +=
    '<canvas id="canvas" width="280" height="280" style="position:absolute;top:100px;left:100px;border:8px solid;"></canvas>';
  document.body.innerHTML +=
    '<img id="canvasimg" style="position:absolute;top:10%;left:52%;width=280px;height=280px;display:none;">';
  document.body.innerHTML +=
    '<input type="button" value="classify" id="sb" size="48" style="position:absolute;top:400px;left:100px;">';
  document.body.innerHTML +=
    '<input type="button" value="clear" id="cb" size="23" style="position:absolute;top:400px;left:180px;">';

  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const rawImage = document.getElementById('canvasimg') as HTMLImageElement;
  const ctx = canvas.getContext('2d')!;
  const pos = {x: 0, y: 0};

  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, 280, 280);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mousedown', setPosition);
  canvas.addEventListener('mouseenter', setPosition);

  const saveButton = document.getElementById('sb') as HTMLButtonElement;
  saveButton.addEventListener('click', classify);
  const clearButton = document.getElementById('cb') as HTMLButtonElement;
  clearButton.addEventListener('click', erase);

  function setPosition(e: MouseEvent) {
    pos.x = e.clientX - 100;
    pos.y = e.clientY - 100;
  }

  function draw(e: MouseEvent) {
    if (e.buttons != 1) return;
    ctx.beginPath();
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    rawImage.src = canvas.toDataURL('image/png');
  }

  function erase() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 280, 280);
  }

  function classify() {
    const raw = tf.browser.fromPixels(rawImage, 1);
    const resized = tf.image.resizeBilinear(raw, [28, 28]);
    const tensor = resized.expandDims(0);
    const prediction = model.predict(tensor) as tf.Tensor<tf.Rank>;
    const pIndex = tf.argMax(prediction, 1).dataSync();

    alert(pIndex);
  }
}

function getModel(): tf.Sequential {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu'}),
  );
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
  model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function train(model: tf.Sequential, data: MnistData) {
  const metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy'];
  const container = {name: 'Model Training', styles: {height: '640px'}};
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 20,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

async function run() {
  const data = new MnistData();
  await data.load();
  tfvis.show.modelSummary({name: 'Model Architecture'}, model);
  await train(model, data);
  createElements();
  alert('Training is done, try classifying your handwriting!');
}

run();
