const webpack = require('webpack');
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const os = require('os');

const localIP = (() => {
  const ifaces = os.networkInterfaces();
  for (const device of Object.keys(ifaces)) {
    for (const iface of ifaces[device]) {
      if (iface.family !== 'IPv4' || iface.internal !== false) {
        continue;
      }
      if (iface.address.startsWith('192')) {
        return iface.address;
      }
    }
  }
  return 'localhost';
})();

const PORT = 8080;

module.exports = {
  mode: 'development',
  entry: './src/index.ts',
  output: {
    path: path.join(__dirname, './public'),
    filename: '[name].[contenthash].js',
    chunkFilename: '[name].[contenthash].js',
    publicPath: `https://${localIP}:${PORT}/`,
  },
  resolve: {
    modules: [path.resolve(__dirname, './src'), 'node_modules'],
    extensions: ['.tsx', '.ts', '.js', '.json'],
  },

  module: {
    rules: [
      {test: /\.tsx?$/, loader: 'ts-loader'},
      {
        test: /\.js$/,
        enforce: 'pre',
        use: ['source-map-loader'],
      },
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
  devServer: {
    contentBase: path.join(__dirname, './public'),
    watchContentBase: true,
    host: '0.0.0.0',
    port: PORT,
    allowedHosts: [localIP],
    headers: {
      'Access-Control-Allow-Origin': '*',
    },
    historyApiFallback: true,
    // hot: true,
    clientLogLevel: 'error',
    compress: true,
    overlay: true,
    http2: true,
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: path.join(__dirname, './public/index.html'),
    }),
    new webpack.DefinePlugin({
      __DEV__: true,
    }),
  ],
};
