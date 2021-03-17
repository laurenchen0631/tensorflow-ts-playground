import * as tf from '@tensorflow/tfjs';
async function read() {
  const url = '/data/my_model.json';
  const model = await tf.loadLayersModel(url);
  model.summary();
  console.log(model.getWeights());
}

read();
