const tf = require("@tensorflow/tfjs");

// مدل
const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, inputShape: [4], activation: "relu" }));
model.add(tf.layers.dense({ units: 3, activation: "softmax" }));
model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

// دیتابیس
const cpuDatabase = {
  "intel core i7-9700k": [8, 8, 3.6, 12],
  "amd ryzen 5 3600": [6, 12, 3.6, 32],
  "intel core i3-6100": [2, 4, 3.7, 3],
  "intel core i5-10400f": [6, 12, 2.9, 12],
  "intel core i9-9900k": [8, 16, 3.6, 16],
  "amd ryzen 7 3700x": [8, 16, 3.6, 32],
  "amd ryzen 9 5900x": [12, 24, 3.7, 64],
  "intel core i5-11400f": [6, 12, 2.6, 12],
  "intel core i7-11700k": [8, 16, 3.6, 16],
  "intel core i9-11900k": [8, 16, 3.5, 16],
  "amd ryzen 5 5600x": [6, 12, 3.7, 32],
  "amd ryzen 7 5800x": [8, 16, 3.8, 32],
  "intel core i3-10100": [4, 8, 3.6, 6],
  "intel core i5-10600k": [6, 12, 4.1, 12],
  "amd ryzen 3 3100": [4, 8, 3.6, 16],
  "amd ryzen 3 3300x": [4, 8, 3.8, 16],
  "intel core i9-12900k": [16, 24, 3.2, 30],
  "intel core i7-12700k": [12, 20, 3.6, 25],
  "intel core i5-12600k": [10, 16, 3.7, 20],
  "amd ryzen 7 7700x": [8, 16, 4.5, 32],
  "amd ryzen 9 7950x": [16, 32, 4.5, 64],
  "intel core i5-13400f": [10, 16, 2.5, 20],
  "intel core i7-13700k": [16, 24, 3.4, 30],
  "intel core i9-13900k": [24, 32, 3.0, 36],
};

// دیتای آموزش
const xs = tf.tensor2d([
  [8, 16, 3.6, 32],
  [4, 8, 2.5, 8],
  [2, 4, 1.8, 4],
]);
const ys = tf.tensor2d([
  [0, 0, 1],
  [0, 1, 0],
  [1, 0, 0],
]);

// استخراج اسم CPU
function extractCPU(text) {
  const regex = /(intel|amd)[\w\s\-]+/i;
  const match = text.match(regex);
  return match ? match[0].trim() : null;
}

// پیش‌بینی
async function predictCPU(cpuName) {
  const features = cpuDatabase[cpuName.toLowerCase()];
  if (!features) return console.log("CPU پیدا نشد!");

  const input = tf.tensor2d([features]);
  const prediction = model.predict(input);
  const result = await prediction.array();
  const categories = ["ضعیف", "متوسط", "خوب"];
  const maxIndex = result[0].indexOf(Math.max(...result[0]));
  console.log(`این پردازنده برای گیمینگ: ${categories[maxIndex]}`);
}

// آموزش و تست
(async () => {
  await model.fit(xs, ys, { epochs: 100 });
  const inputText = "I have an Intel Core i7-10510U, can I play games with it?";
  const cpuModel = extractCPU(inputText);
  console.log("مدل CPU: ", cpuModel);
  await predictCPU(cpuModel);
})();
