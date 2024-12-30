import { HfInference } from "@huggingface/inference";
import dotenv from "dotenv";

dotenv.config();
const hf = new HfInference(process.env.SECRET_TOKEN);

const imageURL =
  "https://tse4.mm.bing.net/th?id=OIP.bOAvLe13MzVAeT8MkekYcwHaE8&pid=Api&P=0&h=180";
//   "https://tse3.mm.bing.net/th?id=OIP.sgWvkh1DClGnGZGAAB1TnAHaH0&pid=Api&P=0&h=180";
//   "https://tse4.mm.bing.net/th?id=OIP.O6wVX9TSLCs46Yqk6EUhNwHaE7&pid=Api&P=0&h=180";
//   "https://tse4.mm.bing.net/th?id=OIP.7zY84_qTmfrAGCyXBlnuOwHaFj&pid=Api&P=0&h=180";

const response = await fetch(imageURL);
const blob = await response.blob();

const model = "Salesforce/blip-image-captioning-large";

const result = await hf.imageToText({
  data: blob,
  model,
});

console.log("Result: ", result);
