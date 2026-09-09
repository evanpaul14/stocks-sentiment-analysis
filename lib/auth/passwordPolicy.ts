/** No node:crypto imports here — shared as-is between server routes and the browser for live validation. */
export interface PasswordRequirement {
  id: string;
  label: string;
  test: (password: string) => boolean;
}

export const PASSWORD_REQUIREMENTS: PasswordRequirement[] = [
  { id: "length", label: "At least 8 characters", test: (p) => p.length >= 8 },
  { id: "uppercase", label: "One uppercase letter", test: (p) => /[A-Z]/.test(p) },
  { id: "number", label: "One number", test: (p) => /[0-9]/.test(p) },
  {
    id: "special",
    label: "One special character",
    test: (p) => /[^A-Za-z0-9]/.test(p),
  },
];

export function getFailedPasswordRequirements(password: string): PasswordRequirement[] {
  return PASSWORD_REQUIREMENTS.filter((r) => !r.test(password));
}

export function isPasswordValid(password: string): boolean {
  return getFailedPasswordRequirements(password).length === 0;
}

export const PASSWORD_REQUIREMENTS_MESSAGE =
  "Password must be at least 8 characters and include an uppercase letter, a number, and a special character.";
