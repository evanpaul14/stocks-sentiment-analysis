import YahooFinance from "yahoo-finance2";

declare global {
  var __yahooFinance__: InstanceType<typeof YahooFinance> | undefined;
}

export const yahooFinance =
  globalThis.__yahooFinance__ ??
  new YahooFinance({ suppressNotices: ["yahooSurvey"] });

if (process.env.NODE_ENV !== "production") {
  globalThis.__yahooFinance__ = yahooFinance;
}
