import {
  englishDataset,
  englishRecommendedTransformers,
  RegExpMatcher,
} from "obscenity";

/** Terms from obscenity's English dataset that are slurs / hate speech, as opposed to general profanity. */
const SLURS = new Set([
  "abo",
  "abeed",
  "africoon",
  "arabush",
  "boonga",
  "chingchong",
  "chink",
  "dyke",
  "fag",
  "kike",
  "negro",
  "nigger",
  "retard",
  "spastic",
  "tranny",
]);

/** Sexual terms: blocked outright, like slurs. */
const SEXUAL = new Set([
  "anal",
  "bestiality",
  "blowjob",
  "boob",
  "buttplug",
  "cock",
  "cum",
  "cunt",
  "deepthroat",
  "dick",
  "dildo",
  "doggystyle",
  "double penetration",
  "ejaculate",
  "felch",
  "fellatio",
  "finger bang",
  "fisting",
  "gangbang",
  "handjob",
  "hentai",
  "hooker",
  "incest",
  "jerk off",
  "jizz",
  "lubejob",
  "masturbate",
  "orgasm",
  "orgy",
  "penis",
  "porn",
  "pussy",
  "rape",
  "scat",
  "semen",
  "sex",
  "slut",
  "tit",
  "twat",
  "vagina",
  "wank",
  "whore",
]);

const matcher = new RegExpMatcher({
  ...englishDataset.build(),
  ...englishRecommendedTransformers,
});

export type ObscenityLevel = "clean" | "profanity" | "blocked";

/** "blocked" if any slur or sexual term matches, else "profanity" if any other obscenity matches. */
export function classifyObscenity(text: string): ObscenityLevel {
  let level: ObscenityLevel = "clean";
  for (const match of matcher.getAllMatches(text)) {
    const word = englishDataset.getPayloadWithPhraseMetadata(match).phraseMetadata
      ?.originalWord;
    if (word && (SLURS.has(word) || SEXUAL.has(word))) return "blocked";
    level = "profanity";
  }
  return level;
}
