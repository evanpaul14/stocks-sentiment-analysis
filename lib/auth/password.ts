import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";

const SALT_BYTES = 16;
const KEY_LENGTH = 64;
const SCRYPT_PARAMS = { N: 16384, r: 8, p: 1 };

export function hashPassword(plaintext: string): {
  hash: string;
  salt: string;
} {
  const salt = randomBytes(SALT_BYTES).toString("hex");
  const hash = scryptSync(plaintext, salt, KEY_LENGTH, SCRYPT_PARAMS).toString(
    "hex"
  );
  return { hash, salt };
}

export function verifyPassword(
  plaintext: string,
  hash: string,
  salt: string
): boolean {
  const expected = Buffer.from(hash, "hex");
  const actual = scryptSync(plaintext, salt, KEY_LENGTH, SCRYPT_PARAMS);
  if (expected.length !== actual.length) return false;
  return timingSafeEqual(expected, actual);
}
