import { NextResponse, type NextRequest } from "next/server";
import { getLimiter } from "./tokenBucket";

export interface RateLimitOptions {
  /** Unique name for this route's quota bucket. */
  routeName: string;
  limit: number;
  windowMs: number;
}

function getClientIp(request: NextRequest): string {
  const forwardedFor = request.headers.get("x-forwarded-for");
  if (forwardedFor) return forwardedFor.split(",")[0].trim();
  return "unknown";
}

/**
 * Wraps a route handler with an in-memory per-IP token bucket. The reverse
 * proxy in front of this process (nginx/Caddy) must set x-forwarded-for.
 */
export function withRateLimit(
  options: RateLimitOptions,
  handler: (request: NextRequest, context: unknown) => Promise<Response>
) {
  const limiter = getLimiter(options.routeName, options.limit, options.windowMs);

  return async (request: NextRequest, context: unknown): Promise<Response> => {
    const key = `${options.routeName}:${getClientIp(request)}`;
    if (!limiter.consume(key)) {
      return NextResponse.json(
        { error: "Too many requests" },
        { status: 429 }
      );
    }
    return handler(request, context);
  };
}
