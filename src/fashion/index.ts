import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {FMnistData} from './data';

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
    const classNames = [
      'T-shirt/top',
      'Trouser',
      'Pullover',
      'Dress',
      'Coat',
      'Sandal',
      'Shirt',
      'Sneaker',
      'Bag',
      'Ankle boot',
    ];

    alert(classNames[pIndex as any]);
  }
}

function getModel(): tf.Sequential {
  // In the space below create a convolutional neural network that can classify the
  // images of articles of clothing in the Fashion MNIST dataset. Your convolutional
  // neural network should only use the following layers: conv2d, maxPooling2d,
  // flatten, and dense. Since the Fashion MNIST has 10 classes, your output layer
  // should have 10 units and a softmax activation function. You are free to use as
  // many layers, filters, and neurons as you like.
  // HINT: Take a look at the MNIST example.
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 6, filters: 34, activation: 'relu'}),
  );
  model.add(tf.layers.maxPooling2d({poolSize: 2}));

  model.add(tf.layers.flatten());
  // model.add(tf.layers.dense({units: 200, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 100, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  // Compile the model using the categoricalCrossentropy loss,
  // the tf.train.adam() optimizer, and accuracy for your metrics.
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function train(model: tf.Sequential, data: FMnistData) {
  // Set the following metrics for the callback: 'loss', 'val_loss', 'accuracy', 'val_accuracy'.

  // Create the container for the callback. Set the name to 'Model Training' and
  // use a height of 1000px for the styles.

  // Use tfvis.show.fitCallbacks() to setup the callbacks.
  // Use the container and metrics defined above as the parameters.
  const fitCallbacks = tfvis.show.fitCallbacks(
    {name: 'Model Training', styles: {height: '1000px'}},
    ['loss', 'acc'],
    // ['loss', 'val_loss', 'acc', 'val_acc'],
  );

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 6000;
  const TEST_DATA_SIZE = 1000;

  // Get the training batches and resize them. Remember to put your code
  // inside a tf.tidy() clause to clean up all the intermediate tensors.
  // HINT: Take a look at the MNIST example.
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  // Get the testing batches and resize them. Remember to put your code
  // inside a tf.tidy() clause to clean up all the intermediate tensors.
  // HINT: Take a look at the MNIST example.
  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

async function run() {
  const data = new FMnistData();
  await data.load();
  tfvis.show.modelSummary({name: 'Model Architecture'}, model);
  await train(model, data);
  await model.save('downloads://my_model');
  createElements();
  // alert('Training is done, try classifying your handwriting!');
}

run();
