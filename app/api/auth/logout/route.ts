import { NextResponse, type NextRequest } from "next/server";
import { revokeSessionForRequest, clearSessionCookies } from "@/lib/auth/session";

export async function POST(request: NextRequest) {
  await revokeSessionForRequest(request);
  const response = new NextResponse(null, { status: 204 });
  clearSessionCookies(response);
  return response;
}
