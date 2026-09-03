/**
 * NYSE trading-day calendar, hand-maintained. No actively-maintained npm
 * package covers this (the one candidate, `nyse-holidays`, hasn't been
 * updated since 2022) — but NYSE's holiday rules are a small, well-defined,
 * low-maintenance ruleset (~9 observed holidays/year), so a rule table is a
 * reasonable alternative to a dependency. Does not account for early-close
 * half-days (not needed for the daily-trading-day gate this supports).
 */

function nthWeekdayOfMonth(year: number, month: number, weekday: number, n: number): Date {
  const first = new Date(Date.UTC(year, month, 1));
  const firstWeekday = first.getUTCDay();
  const offset = (weekday - firstWeekday + 7) % 7;
  const day = 1 + offset + (n - 1) * 7;
  return new Date(Date.UTC(year, month, day));
}

function lastWeekdayOfMonth(year: number, month: number, weekday: number): Date {
  const lastDay = new Date(Date.UTC(year, month + 1, 0));
  const lastWeekday = lastDay.getUTCDay();
  const offset = (lastWeekday - weekday + 7) % 7;
  return new Date(Date.UTC(year, month, lastDay.getUTCDate() - offset));
}

/** Anonymous Gregorian algorithm for the date of Easter Sunday. */
function easterSunday(year: number): Date {
  const a = year % 19;
  const b = Math.floor(year / 100);
  const c = year % 100;
  const d = Math.floor(b / 4);
  const e = b % 4;
  const f = Math.floor((b + 8) / 25);
  const g = Math.floor((b - f + 1) / 3);
  const h = (19 * a + b - d - g + 15) % 30;
  const i = Math.floor(c / 4);
  const k = c % 4;
  const l = (32 + 2 * e + 2 * i - h - k) % 7;
  const m = Math.floor((a + 11 * h + 22 * l) / 451);
  const month = Math.floor((h + l - 7 * m + 114) / 31) - 1;
  const day = ((h + l - 7 * m + 114) % 31) + 1;
  return new Date(Date.UTC(year, month, day));
}

/** NYSE's "observed" rule: Saturday holidays move to the preceding Friday, Sunday to the following Monday. */
function observedDate(date: Date): Date {
  const weekday = date.getUTCDay();
  if (weekday === 6) return addDays(date, -1);
  if (weekday === 0) return addDays(date, 1);
  return date;
}

function addDays(date: Date, days: number): Date {
  const result = new Date(date);
  result.setUTCDate(result.getUTCDate() + days);
  return result;
}

function toDateKey(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function nyseHolidaysForYear(year: number): Set<string> {
  const easter = easterSunday(year);
  const goodFriday = addDays(easter, -2);

  const holidays = [
    observedDate(new Date(Date.UTC(year, 0, 1))), // New Year's Day
    nthWeekdayOfMonth(year, 0, 1, 3), // MLK Day — 3rd Monday of January
    nthWeekdayOfMonth(year, 1, 1, 3), // Washington's Birthday — 3rd Monday of February
    goodFriday,
    lastWeekdayOfMonth(year, 4, 1), // Memorial Day — last Monday of May
    observedDate(new Date(Date.UTC(year, 5, 19))), // Juneteenth
    observedDate(new Date(Date.UTC(year, 6, 4))), // Independence Day
    nthWeekdayOfMonth(year, 8, 1, 1), // Labor Day — 1st Monday of September
    nthWeekdayOfMonth(year, 10, 4, 4), // Thanksgiving — 4th Thursday of November
    observedDate(new Date(Date.UTC(year, 11, 25))), // Christmas Day
  ];

  return new Set(holidays.map(toDateKey));
}

const holidayCacheByYear = new Map<number, Set<string>>();

function getHolidaysForYear(year: number): Set<string> {
  let cached = holidayCacheByYear.get(year);
  if (!cached) {
    cached = nyseHolidaysForYear(year);
    holidayCacheByYear.set(year, cached);
  }
  return cached;
}

/** date is a civil date in a "YYYY-MM-DD" string or a Date (read as UTC). */
export function isNyseTradingDay(date: Date | string): boolean {
  const d = typeof date === "string" ? new Date(`${date}T00:00:00Z`) : date;
  const weekday = d.getUTCDay();
  if (weekday === 0 || weekday === 6) return false;

  const year = d.getUTCFullYear();
  return !getHolidaysForYear(year).has(toDateKey(d));
}
