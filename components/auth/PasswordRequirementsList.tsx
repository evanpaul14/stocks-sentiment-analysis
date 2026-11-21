import { Check, X } from "lucide-react";
import { PASSWORD_REQUIREMENTS } from "@/lib/auth/passwordPolicy";

/**
 * Pass `id` and point the password input's aria-describedby at it, so screen
 * readers read the rules (and which are met) when the field is focused.
 */
export function PasswordRequirementsList({ password, id }: { password: string; id?: string }) {
  return (
    <ul id={id} aria-label="Password requirements" className="space-y-1 text-xs">
      {PASSWORD_REQUIREMENTS.map((req) => {
        const met = req.test(password);
        return (
          <li
            key={req.id}
            className={`flex items-center gap-1.5 transition-colors ${
              met ? "text-[var(--color-chart-1)]" : "text-muted-foreground"
            }`}
          >
            {met ? (
              <Check className="size-3.5" aria-hidden="true" />
            ) : (
              <X className="size-3.5" aria-hidden="true" />
            )}
            {req.label}
            <span className="sr-only">{met ? " (met)" : " (not met)"}</span>
          </li>
        );
      })}
    </ul>
  );
}
