const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Canvas setup
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 12;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "white";

let drawing = false;
let lastX = 0;
let lastY = 0;

// Mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

// Touch events for mobile
canvas.addEventListener("touchstart", handleTouch);
canvas.addEventListener("touchmove", handleTouch);
canvas.addEventListener("touchend", stopDrawing);

function handleTouch(e) {
  e.preventDefault();
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  
  if (e.type === "touchstart") {
    startDrawing({ clientX: x, clientY: y });
  } else if (e.type === "touchmove") {
    draw({ clientX: x, clientY: y });
  }
}

function startDrawing(e) {
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  lastX = e.clientX - rect.left;
  lastY = e.clientY - rect.top;
}

function draw(e) {
  if (!drawing) return;
  
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  
  lastX = x;
  lastY = y;
}

function stopDrawing() {
  drawing = false;
}

function clearCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").innerText = "";
}

// Simple preprocessing function that matches training data
function preprocessImage() {
  // Create temporary canvas for processing
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext("2d");
  
  // Fill with white background
  tempCtx.fillStyle = "white";
  tempCtx.fillRect(0, 0, 28, 28);
  
  // Draw the original canvas content (simple resize)
  tempCtx.drawImage(canvas, 0, 0, 28, 28);
  
  // Get image data
  const imgData = tempCtx.getImageData(0, 0, 28, 28);
  const data = imgData.data;
  
  // Convert to normalized tensor (same as training)
  const input = new Float32Array(1 * 1 * 28 * 28);
  const mean = 0.1307;
  const std = 0.3081;
  
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const idx = (y * 28 + x) * 4;
      const gray = data[idx]; // White background, black digit
      const normalized = (gray / 255.0 - mean) / std;
      input[y * 28 + x] = normalized;
    }
  }
  
  return input;
}

async function predict() {
  const input = preprocessImage();
  
  if (!input) {
    document.getElementById("result").innerText = "Please draw a digit first";
    return;
  }
  
  const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);
  
  try {
    const session = await ort.InferenceSession.create("mnist.onnx");
    const output = await session.run({ input: tensor });
    const logits = output.output.data;
    
    if (!logits || logits.length === 0) {
      throw new Error("No prediction data received");
    }
    
    // Convert logits to probabilities using softmax
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
    const sumExpLogits = expLogits.reduce((sum, exp) => sum + exp, 0);
    const probabilities = expLogits.map(exp => exp / sumExpLogits);
    
    // Get top prediction
    const predictions = [];
    for (let i = 0; i < probabilities.length; i++) {
      predictions.push({ digit: i, probability: probabilities[i] });
    }
    predictions.sort((a, b) => b.probability - a.probability);
    
    const topPrediction = predictions[0];
    
    if (!topPrediction || typeof topPrediction.digit === 'undefined') {
      throw new Error("Invalid prediction structure");
    }
    
    const confidence = topPrediction.probability;
    
    let resultText = `Predicted: ${topPrediction.digit} (${(confidence * 100).toFixed(1)}%)`;
       
    document.getElementById("result").innerText = resultText;
    
  } catch (err) {
    console.error("Inference error:", err);
    console.error("Error details:", err.stack);
    document.getElementById("result").innerText = "Error: " + err.message;
  }
}


