import brain, { INeuralNetworkOptions, IRNNDefaultOptions } from 'brain.js';

const trainingData = [
  {input: [0, 0], output: [0]},
  {input: [0, 1], output: [1]},
  {input: [1, 0], output: [1]},
  {input: [1, 1], output: [0]}
];

// provide optional config object (or undefined). Defaults shown.
const config: INeuralNetworkOptions = {
  binaryThresh: 0.5,
  hiddenLayers: [3], // array of ints for the sizes of the hidden layers in the network
  activation: 'sigmoid' // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh'],
};

// create a simple feed forward neural network with backpropagation
const weakNN = new brain.NeuralNetwork(config);

weakNN.train(trainingData);
// const weakOutput = weakNN.run([1, 0]); // [0.987]

const medConfig: IRNNDefaultOptions = {
  inputSize: 2, //or
  // inputRange: 4, //or 2
  hiddenLayers: [4, 4], //or [4]?
  outputSize: 2, //or 4?
  // learningRate: .01,
  // decayRate: .999
}

const medNN = new brain.NeuralNetwork(medConfig);

medNN.train(trainingData);

const stgConfig: IRNNDefaultOptions = {
  inputSize: 20,
  // inputRange: 20,
  hiddenLayers: [20, 20],
  outputSize: 20,
  // learningRate: 0.01,
  // decayRate: 0.999,
};

const stgNN = new brain.NeuralNetwork(stgConfig);

stgNN.train(trainingData);

const r = (n: number) => Math.round(n * 1000) / 1000;
const score = {
  weak: 0,
  med: 0,
  stg: 0
}

trainingData.forEach(data => {
  console.log(`\t${data.input.join('^')}=${data.output}`);
  const w = weakNN.run(data.input);
  const rW = Math.round(w[0]);
  const m = medNN.run(data.input);
  const rM = Math.round(m[0]);
  const s = stgNN.run(data.input);
  const rS = Math.round(s[0]);
  console.log('weakNN:', w, '~=', r(w[0]), '=>', rW);
  console.log('medNN:', m, '~=', r(m[0]), '=>', rM);
  console.log('stgNN:', s, '~=', r(s[0]), '=>', rS);
  if (rW === data.output[0]) score.weak++;
  if (rM === data.output[0]) score.med++;
  if (rS === data.output[0]) score.stg++;
});

console.log(`\nNN\tPoints\nScore
Weak\t${score.weak}\t${score.weak / trainingData.length}
Medium\t${score.med}\t${score.med / trainingData.length}
Strong\t${score.stg}\t${score.stg / trainingData.length}`);