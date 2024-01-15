// Initialize Image Classifier with MobileNet.
const classifier = ml5.imageClassifier("MobileNet");
classifier.classify(document.getElementById("image"), gotResult);

// Function to run when results arrive
function gotResult(error, results) {
  const element = document.getElementById("result");
  if (error) {
    element.innerHTML = error;
  } else {
    let num = results[0].confidence * 100;
    element.innerHTML =
      results[0].label + "<br>Confidence: " + num.toFixed(2) + "%";
  }
}
