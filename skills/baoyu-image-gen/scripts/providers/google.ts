import path from "node:path";
import { readFile } from "node:fs/promises";
import type { CliArgs } from "../types";

const GOOGLE_MULTIMODAL_MODELS = ["gemini-3-pro-image-preview"];
const GOOGLE_IMAGEN_MODELS = ["imagen-3.0-generate-002", "imagen-3.0-generate-001"];

export function getDefaultModel(): string {
  return process.env.GOOGLE_IMAGE_MODEL || "gemini-3-pro-image-preview";
}

function isGoogleMultimodal(model: string): boolean {
  return GOOGLE_MULTIMODAL_MODELS.some((m) => model.includes(m));
}

function isGoogleImagen(model: string): boolean {
  return GOOGLE_IMAGEN_MODELS.some((m) => model.includes(m));
}

function getGoogleApiKey(): string | null {
  return process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || null;
}

function getGoogleImageSize(args: CliArgs): "1K" | "2K" | "4K" {
  if (args.imageSize) return args.imageSize as "1K" | "2K" | "4K";
  return args.quality === "2k" ? "2K" : "1K";
}

function buildPromptWithAspect(prompt: string, ar: string | null, quality: CliArgs["quality"]): string {
  let result = prompt;
  if (ar) {
    result += ` Aspect ratio: ${ar}.`;
  }
  if (quality === "2k") {
    result += " High resolution 2048px.";
  }
  return result;
}

async function readImageAsBase64(p: string): Promise<{ data: string; mimeType: string }> {
  const buf = await readFile(p);
  const ext = path.extname(p).toLowerCase();
  let mimeType = "image/png";
  if (ext === ".jpg" || ext === ".jpeg") mimeType = "image/jpeg";
  else if (ext === ".gif") mimeType = "image/gif";
  else if (ext === ".webp") mimeType = "image/webp";
  return { data: buf.toString("base64"), mimeType };
}

async function generateWithGemini(
  prompt: string,
  model: string,
  args: CliArgs
): Promise<Uint8Array> {
  const { GoogleGenAI } = await import("@google/genai");

  const apiKey = getGoogleApiKey();
  if (!apiKey) throw new Error("GOOGLE_API_KEY or GEMINI_API_KEY is required");

  const ai = new GoogleGenAI({
    apiKey,
    httpOptions: {
      baseUrl: process.env.GOOGLE_BASE_URL || undefined,
    },
  });

  const input: Array<{ type: "text" | "image"; text?: string; data?: string; mime_type?: string }> = [];
  for (const refPath of args.referenceImages) {
    const { data, mimeType } = await readImageAsBase64(refPath);
    input.push({ type: "image", data, mime_type: mimeType });
  }
  input.push({ type: "text", text: prompt });

  const imageConfig: { image_size: "1K" | "2K" | "4K"; aspect_ratio?: string } = {
    image_size: getGoogleImageSize(args),
  };
  if (args.aspectRatio) {
    imageConfig.aspect_ratio = args.aspectRatio;
  }

  console.log("Generating image with Gemini...", imageConfig);
  const interaction = await ai.interactions.create({
    model,
    input,
    response_modalities: ["image"],
    generation_config: {
      image_config: imageConfig,
    },
  });
  console.log("Generation completed.");

  for (const output of interaction.outputs || []) {
    if (output.type === "image" && output.data) {
      return Uint8Array.from(Buffer.from(output.data, "base64"));
    }
  }

  throw new Error("No image in response");
}

async function generateWithImagen(
  prompt: string,
  model: string,
  args: CliArgs
): Promise<Uint8Array> {
  const { experimental_generateImage: generateImage } = await import("ai");
  const { createGoogleGenerativeAI } = await import("@ai-sdk/google");

  const google = createGoogleGenerativeAI({
    apiKey: getGoogleApiKey() || undefined,
    baseURL: process.env.GOOGLE_BASE_URL,
  });

  const fullPrompt = buildPromptWithAspect(prompt, args.aspectRatio, args.quality);

  const result = await generateImage({
    model: google.image(model),
    prompt: fullPrompt,
    n: args.n,
    aspectRatio: args.aspectRatio || undefined,
  });

  const img = result.images[0];
  if (!img) throw new Error("No image in response");

  if (img.uint8Array) return img.uint8Array;
  if (img.base64) return Uint8Array.from(Buffer.from(img.base64, "base64"));

  throw new Error("Cannot extract image data");
}

export async function generateImage(
  prompt: string,
  model: string,
  args: CliArgs
): Promise<Uint8Array> {
  if (isGoogleImagen(model)) {
    if (args.referenceImages.length > 0) {
      console.error("Warning: Reference images not supported with Imagen models, ignoring.");
    }
    return generateWithImagen(prompt, model, args);
  }

  if (!isGoogleMultimodal(model) && args.referenceImages.length > 0) {
    console.error("Warning: Reference images are only supported with Gemini multimodal models.");
  }

  return generateWithGemini(prompt, model, args);
}
