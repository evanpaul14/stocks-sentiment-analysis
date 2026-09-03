import fs from "node:fs";
import path from "node:path";

/** Writes the IndexNow verification file (public/{key}.txt) at startup if configured. */
export function writeIndexNowKeyFile(): void {
  const key = process.env.INDEXNOW_KEY;
  if (!key) return;

  const filePath = path.join(process.cwd(), "public", `${key}.txt`);
  if (fs.existsSync(filePath)) return;

  fs.writeFileSync(filePath, key);
  console.log(`[indexnow] wrote verification file: public/${key}.txt`);
}
