import * as tf from '@tensorflow/tfjs';

async function run() {
  const MODEL_URL = '/data/linear/model.json';
  const model = await tf.loadLayersModel(MODEL_URL);
  console.log(model.summary());
  const input = tf.tensor2d([10.0], [1, 1]);
  const result = model.predict(input);
  alert(result);
}
run();
