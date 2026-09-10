const EASTERN_TZ = "America/New_York";

/** Today's civil date in America/New_York, as "YYYY-MM-DD". */
export function todayInEastern(now: Date = new Date()): string {
  const formatter = new Intl.DateTimeFormat("en-CA", {
    timeZone: EASTERN_TZ,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  return formatter.format(now); // en-CA formats as YYYY-MM-DD
}

/** Current hour/minute in America/New_York (24h). */
export function currentEasternTime(now: Date = new Date()): { hour: number; minute: number } {
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone: EASTERN_TZ,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  const parts = formatter.formatToParts(now);
  const hour = Number(parts.find((p) => p.type === "hour")?.value ?? "0");
  const minute = Number(parts.find((p) => p.type === "minute")?.value ?? "0");
  return { hour: hour === 24 ? 0 : hour, minute };
}

/** Human-readable date label for market summary titles/prompts, e.g. "August 21, 2026". */
export function formatDateLabel(dateKey: string): string {
  const date = new Date(`${dateKey}T00:00:00Z`);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(date);
}

/** Weekday-inclusive date label, e.g. "Friday, July 17, 2026" — used in SEO titles to match how people search. */
export function formatDateLabelWithWeekday(dateKey: string): string {
  const date = new Date(`${dateKey}T00:00:00Z`);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(date);
}

/** Label for the Monday of the current week (Eastern time), e.g. "August 10, 2026". Used on weekly recap pages. */
export function formatWeekOfLabel(now: Date = new Date()): string {
  const todayKey = todayInEastern(now);
  const date = new Date(`${todayKey}T00:00:00Z`);
  const day = date.getUTCDay(); // 0 = Sunday
  const diffToMonday = day === 0 ? -6 : 1 - day;
  date.setUTCDate(date.getUTCDate() + diffToMonday);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(date);
}
