export async function register() {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;

  await import("./lib/db/client"); // opens the connection, runs pending migrations

  // Production relies on the `prebuild` npm script to write this file
  // before `next build` runs — `next start` serves a public/ snapshot
  // taken at build time, so writing it here would be too late. This call
  // only matters for `next dev`, which serves public/ dynamically.
  const { writeIndexNowKeyFile } = await import("./lib/seo/writeIndexNowKey");
  writeIndexNowKeyFile();

  const { startScheduler } = await import("./lib/cron/scheduler");
  startScheduler();
}
