import { NextResponse, type NextRequest } from "next/server";
import {
  exchangeCodeForIdToken,
  verifyAndDecodeIdToken,
  OAUTH_STATE_COOKIE,
  OAUTH_VERIFIER_COOKIE,
} from "@/lib/auth/google";
import { createSession, applySessionCookies } from "@/lib/auth/session";
import * as users from "@/lib/db/queries/users";

export const dynamic = "force-dynamic";

// Built from SITE_BASE_URL, not request.url — behind the Caddy reverse
// proxy, request.url reflects the internal localhost origin Node is bound
// to, not the public domain (see how sitemap.ts/robots.ts/mailgun.ts do this).
function absoluteUrl(path: string): URL {
  return new URL(path, process.env.SITE_BASE_URL ?? "http://localhost:3000");
}

function failureRedirect(reason: string) {
  const url = absoluteUrl("/login");
  url.searchParams.set("error", reason);
  return NextResponse.redirect(url);
}

export async function GET(request: NextRequest) {
  const code = request.nextUrl.searchParams.get("code");
  const state = request.nextUrl.searchParams.get("state");
  const expectedState = request.cookies.get(OAUTH_STATE_COOKIE)?.value;
  const codeVerifier = request.cookies.get(OAUTH_VERIFIER_COOKIE)?.value;

  if (!code || !state || !expectedState || !codeVerifier || state !== expectedState) {
    return failureRedirect("google_oauth_failed");
  }

  try {
    const idToken = await exchangeCodeForIdToken(code, codeVerifier);
    const identity = await verifyAndDecodeIdToken(idToken);
    if (!identity.emailVerified) {
      return failureRedirect("google_email_unverified");
    }

    let user = await users.getByGoogleSub(identity.sub);
    if (!user) {
      const existingByEmail = await users.getByEmail(identity.email);
      if (existingByEmail) {
        users.linkGoogleSub(existingByEmail.id, identity.sub);
        user = existingByEmail;
      } else {
        user = (await users.insertWithGoogle(identity.email, identity.sub)) ?? undefined;
      }
    }
    if (!user) {
      return failureRedirect("google_oauth_failed");
    }

    const { raw } = await createSession(user.id);
    const response = NextResponse.redirect(absoluteUrl("/account"));
    applySessionCookies(response, raw);
    response.cookies.delete(OAUTH_STATE_COOKIE);
    response.cookies.delete(OAUTH_VERIFIER_COOKIE);
    return response;
  } catch (error) {
    console.error("[api/auth/google/callback] failed", error);
    return failureRedirect("google_oauth_failed");
  }
}
