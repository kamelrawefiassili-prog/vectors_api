const express = require("express");
const cors = require("cors");
const axios = require("axios");
const sharp = require("sharp");
const FormData = require("form-data");

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

const JINA_API_KEY = process.env.JINA_API_KEY;
const JINA_URL = "https://api.jina.ai/v1/embeddings";
const JINA_MODEL = "jina-clip-v2";

// Background removal service (Modal.com), provided by the user.
// Accepts GET requests with ?url=<image url>, returns raw PNG bytes.
const MODAL_REMOVE_BG_URL = process.env.MODAL_REMOVE_BG_URL ||
  "https://kamelrawefiassili-prog--rembg-api-remove-bg.modal.run/";

// ImgBB, used to host the background-removed image permanently so it
// has a stable URL we can store in the database and re-fetch later
// (both for indexing and for future re-indexing).
const IMGBB_API_KEY = process.env.IMGBB_API_KEY;
const IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload";

/**
 * Inspect a PNG's actual alpha (transparency) channel and compute the
 * fraction of pixels that are meaningfully visible (alpha above a
 * small noise threshold). A genuinely blank/failed crop result is
 * ~100% transparent even when minor compression noise pushes a few
 * pixels' alpha slightly above zero — a byte-size or exact-hash check
 * can miss that noise, but reading the real pixel data cannot.
 * Returns a number from 0 (fully transparent/blank) to 1 (fully opaque).
 */
async function getVisiblePixelFraction(pngBuffer) {
  const { data, info } = await sharp(pngBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const channels = info.channels; // 4 (RGBA) after ensureAlpha()
  const totalPixels = info.width * info.height;
  if (totalPixels === 0) return 0;

  const ALPHA_NOISE_THRESHOLD = 10; // out of 255 — ignores faint compression artifacts
  let visibleCount = 0;

  for (let i = 0; i < data.length; i += channels) {
    const alpha = data[i + 3];
    if (alpha > ALPHA_NOISE_THRESHOLD) {
      visibleCount++;
    }
  }

  return visibleCount / totalPixels;
}

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

app.get("/api/process-and-crop", (req, res) => {
  res.json({
    status: "ok",
    message: "Endpoint is live. Send a POST with { image_url } to remove background and get a permanent hosted URL.",
    modal_url_configured: Boolean(MODAL_REMOVE_BG_URL),
    imgbb_key_configured: Boolean(IMGBB_API_KEY)
  });
});

/**
 * POST /api/process-and-crop
 * Body: { image_url: "<original product image URL>" }
 *
 * Sequence:
 * 1. Call the Modal.com background-removal API with the original URL.
 *    It returns raw PNG bytes (background already removed).
 * 2. Upload those PNG bytes to ImgBB to get a permanent, stable URL.
 * 3. Return that new URL so it can be stored in the database
 *    (e.g. products.cropped_image).
 *
 * This does NOT call Jina — embedding happens separately via
 * /api/get-embedding, using the returned cropped image URL as input.
 */
app.post("/api/process-and-crop", async (req, res) => {
  try {
    const { image_url } = req.body || {};

    if (!image_url) {
      return res.status(400).json({
        status: "error",
        message: "image_url مفقود في جسم الطلب."
      });
    }

    if (!IMGBB_API_KEY) {
      return res.status(500).json({
        status: "error",
        message: "IMGBB_API_KEY غير مضبوط في متغيرات البيئة على Vercel."
      });
    }

    // Collected step-by-step log messages, returned to the client so
    // reindex.php / search.php can display a live, detailed trace of
    // what happened for this specific image — not just the final
    // success/fail outcome.
    const steps = [];
    function logStep(message) {
      steps.push({ time: new Date().toISOString(), message: message });
    }

    logStep(`📥 جلب الصورة الأصلية من: ${image_url}`);

    // Step 1: remove background via Modal.
    let pngBuffer;
    try {
      logStep('✂️ إرسال الصورة إلى Modal لإزالة الخلفية...');
      const modalResponse = await axios.get(MODAL_REMOVE_BG_URL, {
        params: { url: image_url },
        responseType: "arraybuffer",
        // Kept under Vercel's 60s function limit, leaving headroom for
        // the ImgBB upload step that follows in the same request.
        timeout: 45000
      });
      pngBuffer = Buffer.from(modalResponse.data);
      logStep(`📦 تم استلام الناتج من Modal (${pngBuffer.length} بايت).`);

      // A fully (or almost fully) transparent/blank result — which can
      // happen if rembg fails to detect any foreground subject, or if
      // the source image itself failed to load correctly — compresses
      // to a very small PNG. A real product photo with visible content
      // is essentially never this small. Reject it here with a clear
      // error rather than silently returning a blank image that looks
      // like a successful crop.
      const MIN_VALID_PNG_BYTES = 3000;
      if (pngBuffer.length < MIN_VALID_PNG_BYTES) {
        logStep(`❌ الحجم (${pngBuffer.length} بايت) أقل من الحد الأدنى المقبول (${MIN_VALID_PNG_BYTES} بايت).`);
        return res.status(502).json({
          status: "error",
          stage: "background_removal",
          steps: steps,
          message: `نتيجة القص فارغة أو شبه فارغة (${pngBuffer.length} بايت) — يبدو أن عملية إزالة الخلفية لم تتعرف على أي محتوى في الصورة.`
        });
      }

      // Read the actual pixel data and measure how much of the image
      // is genuinely visible (non-transparent). This catches blank or
      // near-blank crop failures reliably, even when minor compression
      // noise means the file isn't tiny and doesn't byte-for-byte match
      // any single "known blank" reference image.
      logStep('🔍 تحليل الصورة على مستوى البكسل للتحقق من وجود محتوى مرئي...');
      try {
        const visibleFraction = await getVisiblePixelFraction(pngBuffer);
        const visiblePercent = Math.round(visibleFraction * 1000) / 10;
        logStep(`📊 نسبة البكسلات المرئية في الصورة: ${visiblePercent}%.`);

        const MIN_VISIBLE_FRACTION = 0.01; // at least 1% of pixels must be visible
        if (visibleFraction < MIN_VISIBLE_FRACTION) {
          logStep(`❌ نسبة المحتوى المرئي (${visiblePercent}%) أقل من الحد الأدنى المقبول (1%) — الصورة شبه فارغة.`);
          return res.status(502).json({
            status: "error",
            stage: "background_removal",
            steps: steps,
            message: `نتيجة القص شبه فارغة (نسبة المحتوى المرئي: ${visiblePercent}%) — يبدو أن عملية إزالة الخلفية لم تتعرف على أي محتوى حقيقي في الصورة.`
          });
        }
        logStep('✅ تم العثور على محتوى مرئي كافٍ — يبدو أن القص نجح.');
      } catch (pixelErr) {
        // If pixel analysis itself fails for any reason (corrupt PNG,
        // unexpected format), don't silently accept a possibly-blank
        // result — treat it as a failed crop, matching the caution of
        // the size check above.
        logStep(`❌ فشل تحليل البكسلات (${pixelErr.message}) — سيتم اعتبار القص فاشلاً تحسباً.`);
        return res.status(502).json({
          status: "error",
          stage: "background_removal",
          steps: steps,
          message: `تعذر التحقق من صحة الصورة الناتجة عن القص: ${pixelErr.message}`
        });
      }
    } catch (modalErr) {
      const statusCode = modalErr.response ? modalErr.response.status : null;
      let detail = modalErr.message;
      // Modal returns JSON error details in the response body on
      // failure; try to surface that instead of a generic axios message.
      if (modalErr.response && modalErr.response.data) {
        try {
          const parsed = JSON.parse(Buffer.from(modalErr.response.data).toString("utf-8"));
          detail = parsed.detail || detail;
        } catch (parseErr) {
          // response wasn't JSON — keep the generic message
        }
      }
      logStep(`❌ فشل الاتصال بـ Modal: ${detail}`);
      logStep(`❌ فشل الاتصال بـ Modal: ${detail}`);
      return res.status(502).json({
        status: "error",
        stage: "background_removal",
        steps: steps,
        message: `فشل قص الخلفية عبر Modal${statusCode ? ` (HTTP ${statusCode})` : ""}: ${detail}`
      });
    }

    // Step 2: upload the resulting PNG to ImgBB for a permanent URL.
    let hostedUrl;
    try {
      logStep('⏫ جاري رفع الصورة المقصوصة إلى ImgBB...');
      const base64Png = pngBuffer.toString("base64");

      const form = new FormData();
      form.append("key", IMGBB_API_KEY);
      form.append("image", base64Png);

      const imgbbResponse = await axios.post(IMGBB_UPLOAD_URL, form, {
        headers: form.getHeaders(),
        timeout: 30000
      });

      if (
        !imgbbResponse.data ||
        !imgbbResponse.data.data ||
        !imgbbResponse.data.data.url
      ) {
        throw new Error("رد ImgBB لا يحتوي على رابط صورة صالح.");
      }

      hostedUrl = imgbbResponse.data.data.url;
      logStep(`✅ تم الرفع بنجاح: ${hostedUrl}`);
    } catch (imgbbErr) {
      const statusCode = imgbbErr.response ? imgbbErr.response.status : null;
      logStep(`❌ فشل الرفع إلى ImgBB: ${imgbbErr.message}`);
      return res.status(502).json({
        status: "error",
        stage: "imgbb_upload",
        steps: steps,
        message: `فشل رفع الصورة المقصوصة إلى ImgBB${statusCode ? ` (HTTP ${statusCode})` : ""}: ${imgbbErr.message}`
      });
    }

    logStep('🎉 اكتملت عملية القص والرفع بنجاح.');

    return res.json({
      status: "success",
      cropped_image_url: hostedUrl,
      steps: steps
    });
  } catch (err) {
    const errorDetails = err.response ? err.response.data : err.message;
    return res.status(500).json({
      status: "error",
      message: typeof errorDetails === "object" ? JSON.stringify(errorDetails) : errorDetails
    });
  }
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

    if (!rawInput) {
      return res.status(400).json({
        status: "error",
        message: "رابط أو بيانات الصورة مفقودة (image أو url)"
      });
    }

    let imagePayload;

    if (rawInput.startsWith("data:image")) {
      imagePayload = { image: rawInput };
    } else {
      try {
        const imgRes = await axios.get(rawInput, {
          responseType: "arraybuffer",
          headers: {
            "User-Agent": "Mozilla/5.0 (compatible; VectorsAPIProxy/1.0)",
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

app.use((req, res) => {
  res.status(404).json({
    status: "error",
    message: `Route not found: ${req.method} ${req.path}`
  });
});

module.exports = app;
