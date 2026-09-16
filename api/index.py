const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const JINA_API_KEY = process.env.JINA_API_KEY || "YOUR_JINA_API_KEY";
const JINA_URL = "https://api.jina.ai/v1/embeddings";

app.get("/", (req, res) => {
  res.send("✅ Vectors API Proxy is running with Express & CORS!");
});

app.post("/api/get-embedding", async (req, res) => {
  try {
    const { url, image } = req.body || {};
    const rawInput = image || url;

    if (!rawInput) {
      return res.status(400).json({ status: "error", message: "رابط أو بيانات الصورة مفقودة" });
    }

    let imagePayload;

    if (rawInput.startsWith("data:image")) {
      imagePayload = { image: rawInput };
    } else {
      try {
        const imgRes = await axios.get(rawInput, {
          responseType: "arraybuffer",
          headers: { "User-Agent": "Mozilla/5.0" },
          timeout: 12000
        });
        const contentType = imgRes.headers["content-type"] || "image/jpeg";
        const base64Str = Buffer.from(imgRes.data, "binary").toString("base64");
        imagePayload = { image: `data:${contentType};base64,${base64Str}` };
      } catch (err) {
        imagePayload = { url: rawInput };
      }
    }

    const jinaRes = await axios.post(
      JINA_URL,
      {
        model: "jina-clip-v1",
        input: [imagePayload]
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${JINA_API_KEY}`
        },
        timeout: 25000
      }
    );

    if (jinaRes.data && jinaRes.data.data && jinaRes.data.data.length > 0) {
      return res.json({
        status: "success",
        embedding: jinaRes.data.data[0].embedding
      });
    } else {
      return res.status(500).json({
        status: "error",
        message: "استجابة غير متوقعة من Jina AI",
        details: jinaRes.data
      });
    }
  } catch (err) {
    const errorDetails = err.response ? err.response.data : err.message;
    return res.status(500).json({
      status: "error",
      message: typeof errorDetails === "object" ? JSON.stringify(errorDetails) : errorDetails
    });
  }
});

module.exports = app;
