import fs from "node:fs";
import path from "node:path";
import type Database from "better-sqlite3";

const MIGRATIONS_DIR = path.join(process.cwd(), "lib", "db", "migrations");

/** Takes the db instance as a parameter (rather than importing the client
 * singleton) so it can be called from within client.ts itself without a
 * circular import — the DB self-migrates on first connection. */
export function runMigrations(sqlite: Database.Database): void {
  sqlite.exec(`
    CREATE TABLE IF NOT EXISTS _migrations (
      name TEXT PRIMARY KEY,
      applied_at TEXT NOT NULL DEFAULT (current_timestamp)
    );
  `);

  const applied = new Set(
    sqlite
      .prepare<[], { name: string }>("SELECT name FROM _migrations")
      .all()
      .map((row) => row.name)
  );

  const files = fs
    .readdirSync(MIGRATIONS_DIR)
    .filter((f) => f.endsWith(".sql"))
    .sort();

  for (const file of files) {
    if (applied.has(file)) continue;

    const sql = fs.readFileSync(path.join(MIGRATIONS_DIR, file), "utf-8");
    const applyMigration = sqlite.transaction(() => {
      sqlite.exec(sql);
      sqlite
        .prepare("INSERT OR IGNORE INTO _migrations (name) VALUES (?)")
        .run(file);
    });

    // Migration bodies use IF NOT EXISTS throughout so a concurrent process
    // (e.g. Next.js build workers opening the same DB file at once) racing
    // this is harmless rather than a fatal "already exists" error.
    applyMigration();
    console.log(`[db] applied migration ${file}`);
  }
}
