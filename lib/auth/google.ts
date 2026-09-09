import { createHash, randomBytes } from "node:crypto";

const AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token";
const TOKENINFO_ENDPOINT = "https://oauth2.googleapis.com/tokeninfo";

export const OAUTH_STATE_COOKIE = "ssa_oauth_state";
export const OAUTH_VERIFIER_COOKIE = "ssa_oauth_verifier";

function redirectUri(): string {
  return `${process.env.SITE_BASE_URL ?? ""}/api/auth/google/callback`;
}

function base64url(input: Buffer): string {
  return input.toString("base64url");
}

export function buildGoogleAuthUrl(): {
  url: string;
  state: string;
  codeVerifier: string;
} {
  const state = base64url(randomBytes(16));
  const codeVerifier = base64url(randomBytes(32));
  const codeChallenge = base64url(createHash("sha256").update(codeVerifier).digest());

  const params = new URLSearchParams({
    client_id: process.env.GOOGLE_OAUTH_CLIENT_ID ?? "",
    redirect_uri: redirectUri(),
    response_type: "code",
    scope: "openid email",
    state,
    code_challenge: codeChallenge,
    code_challenge_method: "S256",
    prompt: "select_account",
  });

  return { url: `${AUTH_ENDPOINT}?${params.toString()}`, state, codeVerifier };
}

export async function exchangeCodeForIdToken(
  code: string,
  codeVerifier: string
): Promise<string> {
  const response = await fetch(TOKEN_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: process.env.GOOGLE_OAUTH_CLIENT_ID ?? "",
      client_secret: process.env.GOOGLE_OAUTH_CLIENT_SECRET ?? "",
      code,
      code_verifier: codeVerifier,
      grant_type: "authorization_code",
      redirect_uri: redirectUri(),
    }),
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) {
    throw new Error(`Google token exchange failed: ${response.status}`);
  }
  const data = await response.json();
  if (typeof data.id_token !== "string") {
    throw new Error("Google token exchange response missing id_token");
  }
  return data.id_token;
}

export interface GoogleIdentity {
  sub: string;
  email: string;
  emailVerified: boolean;
}

/**
 * Validates the id_token via Google's tokeninfo endpoint rather than local
 * JWKS verification — avoids adding a JWT-verification dependency for the
 * one place this app would otherwise need one.
 */
export async function verifyAndDecodeIdToken(idToken: string): Promise<GoogleIdentity> {
  const response = await fetch(
    `${TOKENINFO_ENDPOINT}?id_token=${encodeURIComponent(idToken)}`,
    { signal: AbortSignal.timeout(10_000) }
  );
  if (!response.ok) {
    throw new Error(`Google tokeninfo validation failed: ${response.status}`);
  }
  const data = await response.json();
  if (data.aud !== process.env.GOOGLE_OAUTH_CLIENT_ID) {
    throw new Error("Google id_token audience mismatch");
  }
  if (typeof data.sub !== "string" || typeof data.email !== "string") {
    throw new Error("Google id_token missing sub/email");
  }
  return {
    sub: data.sub,
    email: data.email.toLowerCase(),
    emailVerified: data.email_verified === "true" || data.email_verified === true,
  };
}
