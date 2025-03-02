# ZOX-GEN-AI
This is a website of ZOX GEN AI. This is a website of  AI Image Generation.

/*
 * AI Image Generator - Full Stack Application
 * This file contains the most important parts of the project consolidated for review.
 */

// ===================== SERVER CODE =====================

// server/index.ts - Main server entry point
import express from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";

// Express server setup
const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }
      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }
      log(logLine);
    }
  });
  next();
});

// Server initialization
(async () => {
  const server = await registerRoutes(app);

  // Error handling
  app.use((err, _req, res, _next) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";
    res.status(status).json({ message });
    throw err;
  });

  // Serve static files in production
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // Try multiple ports, starting with 5000
  const tryPorts = [5000, 5001, 5002, 3000];
  
  function startServer(portIndex = 0) {
    if (portIndex >= tryPorts.length) {
      log("Failed to start server: All ports are in use");
      process.exit(1);
      return;
    }

    const port = tryPorts[portIndex];
    server.listen({
      port,
      host: "0.0.0.0",
      reusePort: true,
    }, () => {
      log(`serving on port ${port}`);
    }).on('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        log(`Port ${port} is busy, trying next port...`);
        startServer(portIndex + 1);
      } else {
        log(`Error starting server: ${err.message}`);
        throw err;
      }
    });
  }
  
  startServer();
})();

// server/routes.ts - API Routes
import { createServer } from "http";
import { storage } from "./storage";
import { insertImageSchema, type ModelType } from "@shared/schema";
import { HfInference } from "@huggingface/inference";

// Create HF inference with better error handling
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

// Validate API key at startup
if (!process.env.HUGGINGFACE_API_KEY) {
  console.error("WARNING: HUGGINGFACE_API_KEY is not set! Image generation will fail.");
}

// Model mapping for different styles using only Hugging Face models
const MODEL_MAPPING = {
  realistic: "CompVis/stable-diffusion-v1-4",
  anime: "hakurei/waifu-diffusion",
  photorealistic: "runwayml/stable-diffusion-v1-5",
  cinematic: "dreamlike-art/dreamlike-photoreal-2.0",
  logo: "CompVis/stable-diffusion-v1-4",
  illustration: "runwayml/stable-diffusion-v1-5"
};

export async function registerRoutes(app) {
  app.post("/api/images", async (req, res) => {
    try {
      console.log("Received image generation request:", req.body);
      const imageData = insertImageSchema.parse(req.body);
      const image = await storage.createImage(imageData);

      // Generate images asynchronously
      generateImages(
        image.id,
        imageData.prompt,
        imageData.model,
        imageData.aspectRatio
      ).catch(console.error);

      res.json(image);
    } catch (error) {
      console.error("Error processing image request:", error);
      res.status(400).json({ 
        message: "Invalid request data",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  app.get("/api/images/:id", async (req, res) => {
    const id = parseInt(req.params.id);
    const image = await storage.getImage(id);

    if (!image) {
      res.status(404).json({ message: "Image not found" });
      return;
    }

    res.json(image);
  });

  const httpServer = createServer(app);
  return httpServer;
}

async function generateImages(
  imageId,
  prompt,
  model,
  aspectRatio
) {
  try {
    console.log(`Starting image generation for model: ${model}`);
    
    // Check if API key is available
    if (!process.env.HUGGINGFACE_API_KEY) {
      throw new Error("HUGGINGFACE_API_KEY is not set in environment variables");
    }
    
    const modelId = MODEL_MAPPING[model];
    console.log(`Using model ID: ${modelId}`);
    const imageUrls = [];

    // Generate 4 variations
    for (let i = 0; i < 4; i++) {
      console.log(`Generating image ${i + 1}/4...`);
      try {
        console.log(`Sending request to HuggingFace API with prompt: "${prompt}"`);
        const result = await hf.textToImage({
          model: modelId,
          inputs: prompt,
          parameters: {
            negative_prompt: "blurry, bad quality, distorted, ugly, deformed",
            guidance_scale: 7.5,
            num_inference_steps: 50,
            width: aspectRatio.width,
            height: aspectRatio.height,
          }
        });

        if (!result) {
          throw new Error("Received empty response from HuggingFace API");
        }

        // Convert blob to base64
        const buffer = await result.arrayBuffer();
        const base64 = Buffer.from(buffer).toString('base64');
        const imageUrl = `data:image/jpeg;base64,${base64}`;
        imageUrls.push(imageUrl);
        console.log(`Successfully generated image ${i + 1}, data size: ${base64.length} chars`);
      } catch (error) {
        console.error(`Error generating image ${i + 1}:`, error);
        if (error instanceof Error) {
          console.error(`Error details: ${error.message}`);
          console.error(`Error stack: ${error.stack}`);
        }
        throw error;
      }
    }

    console.log('All images generated successfully, updating status...');
    await storage.updateImageStatus(imageId, "completed", imageUrls);
  } catch (error) {
    console.error("Image generation failed:", error);
    let errorMessage = "Unknown error";
    if (error instanceof Error) {
      errorMessage = `${error.message}\n${error.stack}`;
      console.error(`Error stack: ${error.stack}`);
    }
    await storage.updateImageStatus(
      imageId,
      "failed",
      [],
      errorMessage
    );
  }
}

// ===================== DATABASE SCHEMA =====================

// shared/schema.ts - Database schema and types
import { pgTable, text, serial, integer, boolean, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const aspectRatioSchema = z.object({
  width: z.number().int().positive(),
  height: z.number().int().positive()
});

// Define available models
export const modelSchema = z.enum([
  "realistic",
  "anime",
  "photorealistic",
  "cinematic",
  "logo",
  "illustration"
]);

export const images = pgTable("images", {
  id: serial("id").primaryKey(),
  prompt: text("prompt").notNull(),
  model: text("model", { enum: ["realistic", "anime", "photorealistic", "cinematic", "logo", "illustration"] }).notNull(),
  aspectRatio: jsonb("aspect_ratio").$type().notNull(),
  imageUrls: text("image_urls").array().notNull(),
  status: text("status", { enum: ["pending", "completed", "failed"] }).notNull().default("pending"),
  error: text("error"),
});

export const insertImageSchema = createInsertSchema(images).pick({
  prompt: true,
  model: true,
  aspectRatio: true,
});

// ===================== FRONTEND CODE =====================

// client/src/App.tsx - Main application component
import React from "react";
import { ThemeProvider } from "next-themes";
import { Route, Switch } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import HomePage from "./pages/home";
import ImagePage from "./pages/image";
import { Toaster } from "./components/ui/toaster";

const queryClient = new QueryClient();

export default function App() {
  return (
    <ThemeProvider defaultTheme="system" enableSystem>
      <QueryClientProvider client={queryClient}>
        <Switch>
          <Route path="/" component={HomePage} />
          <Route path="/images/:id" component={ImagePage} />
        </Switch>
        <Toaster />
      </QueryClientProvider>
    </ThemeProvider>
  );
}

// client/src/pages/home.tsx - Home page with image generation form
import React, { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useNavigate } from "wouter";
import MainLayout from "../components/layouts/main-layout";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../components/ui/form";
import { ToggleGroup, ToggleGroupItem } from "../components/ui/toggle-group";
import { aspectRatioSchema, modelSchema } from "../../shared/schema";

const formSchema = z.object({
  prompt: z.string().min(3).max(200),
  model: modelSchema,
  aspectRatio: aspectRatioSchema,
});

export default function HomePage() {
  const [navigate] = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  
  const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      prompt: "",
      model: "realistic",
      aspectRatio: { width: 512, height: 512 },
    },
  });

  async function onSubmit(values) {
    setIsLoading(true);
    try {
      const response = await fetch("/api/images", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      
      if (!response.ok) {
        throw new Error("Failed to generate image");
      }
      
      const data = await response.json();
      navigate(`/images/${data.id}`);
    } catch (error) {
      console.error("Error generating image:", error);
      // Show error toast
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <MainLayout>
      <div className="max-w-2xl mx-auto">
        <h2 className="text-3xl font-bold mb-6">Generate AI Images</h2>
        
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            {/* Prompt input */}
            <FormField
              control={form.control}
              name="prompt"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Prompt</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="Describe the image you want to generate..."
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            {/* Model selection */}
            <FormField
              control={form.control}
              name="model"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Style</FormLabel>
                  <FormControl>
                    <ToggleGroup
                      type="single"
                      value={field.value}
                      onValueChange={field.onChange}
                      className="flex flex-wrap justify-start"
                    >
                      <ToggleGroupItem value="realistic">Realistic</ToggleGroupItem>
                      <ToggleGroupItem value="anime">Anime</ToggleGroupItem>
                      <ToggleGroupItem value="photorealistic">Photorealistic</ToggleGroupItem>
                      <ToggleGroupItem value="cinematic">Cinematic</ToggleGroupItem>
                      <ToggleGroupItem value="logo">Logo</ToggleGroupItem>
                      <ToggleGroupItem value="illustration">Illustration</ToggleGroupItem>
                    </ToggleGroup>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            {/* Aspect ratio selection */}
            <FormField
              control={form.control}
              name="aspectRatio"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Aspect Ratio</FormLabel>
                  <FormControl>
                    <ToggleGroup
                      type="single"
                      value={`${field.value.width}x${field.value.height}`}
                      onValueChange={(value) => {
                        const [width, height] = value.split('x').map(Number);
                        field.onChange({ width, height });
                      }}
                      className="flex flex-wrap justify-start"
                    >
                      <ToggleGroupItem value="512x512">Square</ToggleGroupItem>
                      <ToggleGroupItem value="768x512">Landscape</ToggleGroupItem>
                      <ToggleGroupItem value="512x768">Portrait</ToggleGroupItem>
                    </ToggleGroup>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? "Generating..." : "Generate Images"}
            </Button>
          </form>
        </Form>
      </div>
    </MainLayout>
  );
}

// client/src/pages/image.tsx - Image display page
import React, { useEffect, useState } from "react";
import { useRoute } from "wouter";
import MainLayout from "../components/layouts/main-layout";
import { Button } from "../components/ui/button";
import { Skeleton } from "../components/ui/skeleton";
import { useToast } from "../components/ui/use-toast";

export default function ImagePage() {
  const [, params] = useRoute("/images/:id");
  const [image, setImage] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();
  const imageId = params?.id ? parseInt(params.id) : null;

  useEffect(() => {
    if (!imageId) return;

    const fetchImage = async () => {
      try {
        const response = await fetch(`/api/images/${imageId}`);
        if (!response.ok) {
          throw new Error("Failed to fetch image");
        }
        const data = await response.json();
        setImage(data);
      } catch (error) {
        console.error("Error fetching image:", error);
        toast({
          title: "Error",
          description: "Failed to load image. Please try again.",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchImage();

    // Poll for updates if status is pending
    const interval = setInterval(async () => {
      if (image && image.status !== "pending") {
        clearInterval(interval);
        return;
      }
      fetchImage();
    }, 2000);

    return () => clearInterval(interval);
  }, [imageId, image?.status]);

  return (
    <MainLayout>
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-bold">Generated Images</h2>
          <Button onClick={() => window.history.back()}>Back</Button>
        </div>

        {isLoading ? (
          <div>
            <Skeleton className="h-8 w-64 mb-4" />
            <div className="grid grid-cols-2 gap-4">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className="aspect-square w-full rounded-md" />
              ))}
            </div>
          </div>
        ) : image ? (
          <div>
            <div className="mb-4">
              <p className="text-lg font-medium">Prompt: {image.prompt}</p>
              <p>
                Status:{" "}
                <span
                  className={
                    image.status === "completed"
                      ? "text-green-500"
                      : image.status === "failed"
                      ? "text-red-500"
                      : "text-yellow-500"
                  }
                >
                  {image.status}
                </span>
              </p>
            </div>

            {image.status === "pending" ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 dark:border-gray-100 mb-4"></div>
                <p className="text-lg">Generating your images...</p>
                <p className="text-sm text-muted-foreground mt-2">
                  This may take up to a minute
                </p>
              </div>
            ) : image.status === "failed" ? (
              <div className="text-center py-12 border rounded-md bg-red-50 dark:bg-red-950/20">
                <p className="text-lg text-red-600 dark:text-red-400">
                  Image generation failed
                </p>
                {image.error && (
                  <p className="text-sm mt-2 max-w-md mx-auto">
                    Error: {image.error}
                  </p>
                )}
                <Button
                  onClick={() => window.history.back()}
                  className="mt-4"
                  variant="outline"
                >
                  Try Again
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4">
                {image.imageUrls.map((url, index) => (
                  <div key={index} className="rounded-md overflow-hidden">
                    <img
                      src={url}
                      alt={`Generated image ${index + 1}`}
                      className="w-full h-auto object-cover"
                      loading="lazy"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-lg">Image not found</p>
            <Button onClick={() => window.history.back()} className="mt-4">
              Go Back
            </Button>
          </div>
        )}
      </div>
    </MainLayout>
  );
}
