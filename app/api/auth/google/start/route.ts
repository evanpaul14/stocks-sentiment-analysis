import { NextResponse } from "next/server";
import {
  buildGoogleAuthUrl,
  OAUTH_STATE_COOKIE,
  OAUTH_VERIFIER_COOKIE,
} from "@/lib/auth/google";

export const dynamic = "force-dynamic";

const OAUTH_COOKIE_MAX_AGE = 10 * 60; // 10 minutes — just long enough for the redirect round trip

export async function GET() {
  if (!process.env.GOOGLE_OAUTH_CLIENT_ID) {
    return NextResponse.json({ error: "Google sign-in is not configured" }, { status: 503 });
  }

  const { url, state, codeVerifier } = buildGoogleAuthUrl();
  const response = NextResponse.redirect(url);
  const isProd = process.env.NODE_ENV === "production";
  const cookieOptions = {
    httpOnly: true,
    secure: isProd,
    sameSite: "lax" as const,
    path: "/",
    maxAge: OAUTH_COOKIE_MAX_AGE,
  };
  response.cookies.set(OAUTH_STATE_COOKIE, state, cookieOptions);
  response.cookies.set(OAUTH_VERIFIER_COOKIE, codeVerifier, cookieOptions);
  return response;
}
