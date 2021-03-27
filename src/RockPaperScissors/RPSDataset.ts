import * as tf from '@tensorflow/tfjs';

export default class RPSDataset {
  private labels: number[];
  public xs: tf.Tensor<tf.Rank> | null;
  public ys: tf.Tensor<tf.Rank> | null;

  constructor() {
    this.labels = [];
    this.xs = null;
    this.ys = null;
  }

  addExample(example: tf.Tensor<tf.Rank>, label: number): void {
    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.labels.push(label);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(label);
      oldX.dispose();
    }
  }

  encodeLabels(numClasses: number): void {
    for (let i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        this.ys = tf.keep(
          tf.tidy(() => {
            return tf.oneHot(tf.tensor1d([this.labels[i]!]).toInt(), numClasses);
          }),
        );
      } else {
        const y = tf.tidy(() => {
          return tf.oneHot(tf.tensor1d([this.labels[i]!]).toInt(), numClasses);
        });
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }
}
