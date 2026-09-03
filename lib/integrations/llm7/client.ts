import OpenAI from "openai";

export const llm7Model = process.env.LLM7_MODEL ?? "fast";

export const llm7Client = process.env.LLM7_API_KEY
  ? new OpenAI({
      baseURL: process.env.LLM7_BASE_URL ?? "https://api.llm7.io/v1",
      apiKey: process.env.LLM7_API_KEY,
    })
  : null;
