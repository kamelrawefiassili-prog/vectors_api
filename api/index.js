const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

const JINA_API_KEY = process.env.JINA_API_KEY;
const JINA_URL = "https://api.jina.ai/v1/embeddings";
const JINA_MODEL = "jina-clip-v1";

app.get("/", (req, res) => {
  res.send("✅ Vectors API Proxy is running with Express & CORS!");
});

app.get("/api/get-embedding", (req, res) => {
  res.json({
    status: "ok",
    message: "Endpoint is live. Send a POST with { image } or { url }.",
    jina_api_key_configured: Boolean(JINA_API_KEY)
  });
});

app.post("/api/get-embedding", async (req, res) => {
  try {
    if (!JINA_API_KEY) {
      return res.status(500).json({
        status: "error",
        message:
          "JINA_API_KEY غير مضبوط في متغيرات البيئة على Vercel. أضفه من Settings → Environment Variables ثم أعد النشر."
      });
    }

    const { url, image } = req.body || {};
    const rawInput = image || url;

    // DEBUG: log exactly what arrived from the client.
    console.log("[DEBUG] req.body keys:", Object.keys(req.body || {}));
    console.log("[DEBUG] typeof image:", typeof image, "| length:", image ? image.length : null);
    console.log("[DEBUG] typeof url:", typeof url, "| value:", url);
    console.log("[DEBUG] rawInput starts with:", rawInput ? String(rawInput).substring(0, 60) : null);

    if (!rawInput) {
      return res.status(400).json({
        status: "error",
        message: "رابط أو بيانات الصورة مفقودة (image أو url)"
      });
    }

    let imagePayload;
    let sourceUsed;

    if (rawInput.startsWith("data:image")) {
      imagePayload = { image: rawInput };
      sourceUsed = "data-url-as-is";
    } else {
      try {
        const imgRes = await axios.get(rawInput, {
          responseType: "arraybuffer",
          headers: { "User-Agent": "Mozilla/5.0 (compatible; VectorsAPIProxy/1.0)" },
          timeout: 12000
        });
        const contentType = imgRes.headers["content-type"] || "image/jpeg";
        const base64Str = Buffer.from(imgRes.data, "binary").toString("base64");
        imagePayload = { image: `data:${contentType};base64,${base64Str}` };
        sourceUsed = "fetched-and-converted";
        console.log("[DEBUG] fetched image, contentType:", contentType, "| base64 length:", base64Str.length);
      } catch (fetchErr) {
        imagePayload = { url: rawInput };
        sourceUsed = "fallback-raw-url";
        console.log("[DEBUG] fetch failed, falling back to raw URL. Error:", fetchErr.message);
      }
    }

    console.log("[DEBUG] sourceUsed:", sourceUsed);
    console.log("[DEBUG] imagePayload keys:", Object.keys(imagePayload));
    console.log("[DEBUG] imagePayload.image length:", imagePayload.image ? imagePayload.image.length : null);
    console.log("[DEBUG] imagePayload.url:", imagePayload.url || null);

    const jinaRequestBody = {
      model: JINA_MODEL,
      input: [imagePayload]
    };

    console.log("[DEBUG] Full body being sent to Jina (image truncated):", JSON.stringify({
      model: jinaRequestBody.model,
      input: [{
        image: imagePayload.image ? imagePayload.image.substring(0, 50) + "...[truncated]" : undefined,
        url: imagePayload.url || undefined
      }]
    }));

    const jinaRes = await axios.post(
      JINA_URL,
      jinaRequestBody,
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${JINA_API_KEY}`
        },
        timeout: 25000
      }
    );

    if (jinaRes.data && Array.isArray(jinaRes.data.data) && jinaRes.data.data.length > 0) {
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
    console.log("[DEBUG] Error caught:", JSON.stringify(errorDetails));
    return res.status(500).json({
      status: "error",
      message: typeof errorDetails === "object" ? JSON.stringify(errorDetails) : errorDetails
    });
  }
});

app.use((req, res) => {
  res.status(404).json({
    status: "error",
    message: `Route not found: ${req.method} ${req.path}`
  });
});

module.exports = app;
