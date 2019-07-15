"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const brain_js_1 = __importDefault(require("brain.js"));
const trainingData = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
];
const config = {
    binaryThresh: 0.5,
    hiddenLayers: [3],
    activation: 'sigmoid'
};
const weakNN = new brain_js_1.default.NeuralNetwork(config);
weakNN.train(trainingData);
const medConfig = {
    inputSize: 2,
    hiddenLayers: [4, 4],
    outputSize: 2,
};
const medNN = new brain_js_1.default.NeuralNetwork(medConfig);
medNN.train(trainingData);
const stgConfig = {
    inputSize: 20,
    hiddenLayers: [20, 20],
    outputSize: 20,
};
const stgNN = new brain_js_1.default.NeuralNetwork(stgConfig);
stgNN.train(trainingData);
const r = (n) => Math.round(n * 1000) / 1000;
const score = {
    weak: 0,
    med: 0,
    stg: 0
};
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
    if (rW === data.output[0])
        score.weak++;
    if (rM === data.output[0])
        score.med++;
    if (rS === data.output[0])
        score.stg++;
});
console.log(`\nNN\tPoints\nScore
Weak\t${score.weak}\t${score.weak / trainingData.length}
Medium\t${score.med}\t${score.med / trainingData.length}
Strong\t${score.stg}\t${score.stg / trainingData.length}`);
