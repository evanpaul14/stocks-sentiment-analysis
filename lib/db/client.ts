import path from "node:path";
import fs from "node:fs";
import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import * as schema from "./schema";
import { runMigrations } from "./migrate";

declare global {
  var __sqlite__: Database.Database | undefined;
}

function openDatabase(): Database.Database {
  // Statically scoped to ./data/<filename> so bundlers don't trace the whole
  // project as a dependency of an arbitrary runtime path (only the filename
  // itself is configurable via DATABASE_FILENAME).
  const filename = process.env.DATABASE_FILENAME ?? "app.db";
  const resolved = path.join(process.cwd(), "data", filename);
  fs.mkdirSync(path.join(process.cwd(), "data"), { recursive: true });

  const db = new Database(resolved);
  db.pragma("busy_timeout = 5000");
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  runMigrations(db);
  return db;
}

// Guard against re-opening the file handle on every Next.js dev hot-reload.
const sqlite = globalThis.__sqlite__ ?? openDatabase();
if (process.env.NODE_ENV !== "production") {
  globalThis.__sqlite__ = sqlite;
}

export const db = drizzle(sqlite, { schema });
export { sqlite };
