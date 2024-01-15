// Train your model first
const data = [
  { r: 255, g: 0, b: 0, color: "red-ish" },
  { r: 254, g: 0, b: 0, color: "red-ish" },
  { r: 253, g: 0, b: 0, color: "red-ish" },
  { r: 0, g: 255, b: 0, color: "green-ish" },
  { r: 0, g: 254, b: 0, color: "green-ish" },
  { r: 0, g: 253, b: 0, color: "green-ish" },
  { r: 0, g: 0, b: 255, color: "blue-ish" },
  { r: 0, g: 0, b: 254, color: "blue-ish" },
  { r: 0, g: 0, b: 253, color: "blue-ish" },
];

const options = {
  task: "classification",
  debug: true,
};

const nn = ml5.neuralNetwork(options);

data.forEach((item) => {
  const inputs = {
    r: item.r,
    g: item.g,
    b: item.b,
  };
  const output = {
    color: item.color,
  };

  nn.addData(inputs, output);
});

nn.normalizeData();

const trainingOptions = {
  epochs: 32,
  batchSize: 12,
};
nn.train(trainingOptions, finishedTraining);

function finishedTraining() {
  console.log("training has completed. You can start your job!");
}

document.querySelector(".btn").addEventListener("click", () => {
  const r = document.getElementById("r-code").value;
  const g = document.getElementById("g-code").value;
  const b = document.getElementById("b-code").value;

  document.getElementById(
    "result-box"
  ).style.backgroundColor = `rgb(${r},${g},${b})`;

  const input = {
    r: parseInt(r),
    g: parseInt(g),
    b: parseInt(b),
  };
  nn.classify(input, handleResults);
});

function handleResults(error, result) {
  if (error) {
    console.error(error);
    return;
  }

  const colorResult = result.reduce(function (prev, current) {
    return prev && prev.confidence > current.confidence ? prev : current;
  });

  console.log(colorResult);

  alert(colorResult.label);
}
