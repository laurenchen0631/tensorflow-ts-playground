import '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

function createElements() {
  document.body.innerHTML += '<img id="img" src="/data/coffee.jpeg"></img>';
  document.body.innerHTML +=
    '<div id="output" style="font-family:courier;font-size:24px;height=300px"></div>';
}

createElements();

const img = document.getElementById('img') as HTMLImageElement;
const outp = document.getElementById('output')!;
mobilenet.load().then((model) => {
  model.classify(img).then((predictions) => {
    console.log(predictions);
    for (let i = 0; i < predictions.length; i++) {
      outp.innerHTML += '<br/>' + predictions[i]?.className + ' : ' + predictions[i]?.probability;
    }
  });
});
