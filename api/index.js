const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

const JINA_API_KEY = process.env.JINA_API_KEY;
const JINA_URL = "https://api.jina.ai/v1/embeddings";
const JINA_MODEL = "jina-clip-v2";

app.get("/", (req, res) => {
  res.send("✅ Vectors API Proxy is running with Express & CORS!");
});

// Health check for the actual endpoint — lets you verify the deployment
// (and whether the API key is configured) without sending an image.
app.get("/api/get-embedding", (req, res) => {
  res.json({
    status: "ok",
    message: "Endpoint is live. Send a POST with { image } or { url }.",
    jina_api_key_configured: Boolean(JINA_API_KEY)
  });
});

app.post("/api/get-embedding", async (req, res) => {
  try {
    // Fail loudly and immediately if the API key isn't set, instead of
    // silently sending a placeholder string to Jina AI and getting a
    // confusing 401 back later.
    if (!JINA_API_KEY) {
      return res.status(500).json({
        status: "error",
        message:
          "JINA_API_KEY غير مضبوط في متغيرات البيئة على Vercel. أضفه من Settings → Environment Variables ثم أعد النشر."
      });
    }

    const { url, image } = req.body || {};
    const rawInput = image || url;

    if (!rawInput) {
      return res.status(400).json({
        status: "error",
        message: "رابط أو بيانات الصورة مفقودة (image أو url)"
      });
    }

    let imagePayload;

    if (rawInput.startsWith("data:image")) {
      // Already a base64 data URL — send as-is.
      imagePayload = { image: rawInput };
    } else {
      // Fetch the URL server-side and convert to base64 before
      // forwarding to Jina AI. We deliberately do NOT fall back to
      // sending the raw URL to Jina AI on fetch failure: Jina AI's
      // own attempt to fetch a blocked/unreachable URL doesn't return
      // a clear error — it returns a confusing schema-validation error
      // ("input should be a valid string", etc) that looks like a bug
      // in our request format when it's actually an unreachable image.
      // Returning our own clear error here instead is far easier to
      // diagnose.
      try {
        const imgRes = await axios.get(rawInput, {
          responseType: "arraybuffer",
          headers: {
            "User-Agent": "Mozilla/5.0 (compatible; VectorsAPIProxy/1.0)",
            // Some image hosts (ImgBB included) reject requests with no
            // Accept header or an unusual one.
            "Accept": "image/*,*/*;q=0.8"
          },
          timeout: 15000,
          maxRedirects: 5
        });
        const contentType = imgRes.headers["content-type"] || "image/jpeg";
        const base64Str = Buffer.from(imgRes.data, "binary").toString("base64");
        imagePayload = { image: `data:${contentType};base64,${base64Str}` };
      } catch (fetchErr) {
        const statusCode = fetchErr.response ? fetchErr.response.status : null;
        return res.status(502).json({
          status: "error",
          message:
            "تعذر جلب الصورة من الرابط المرسل" +
            (statusCode ? ` (HTTP ${statusCode})` : ` (${fetchErr.message})`) +
            `. الرابط: ${rawInput}`
        });
      }
    }

    const jinaRes = await axios.post(
      JINA_URL,
      {
        model: JINA_MODEL,
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
    return res.status(500).json({
      status: "error",
      message: typeof errorDetails === "object" ? JSON.stringify(errorDetails) : errorDetails
    });
  }
});

// Catch-all 404 so unknown routes return a clear JSON error instead of
// an opaque platform-level failure.
app.use((req, res) => {
  res.status(404).json({
    status: "error",
    message: `Route not found: ${req.method} ${req.path}`
  });
});

module.exports = app;
