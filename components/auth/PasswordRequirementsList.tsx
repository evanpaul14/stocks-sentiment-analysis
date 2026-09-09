import { Check, X } from "lucide-react";
import { PASSWORD_REQUIREMENTS } from "@/lib/auth/passwordPolicy";

export function PasswordRequirementsList({ password }: { password: string }) {
  return (
    <ul className="space-y-1 text-xs">
      {PASSWORD_REQUIREMENTS.map((req) => {
        const met = req.test(password);
        return (
          <li
            key={req.id}
            className={`flex items-center gap-1.5 transition-colors ${
              met ? "text-[var(--color-chart-1)]" : "text-muted-foreground"
            }`}
          >
            {met ? <Check className="size-3.5" /> : <X className="size-3.5" />}
            {req.label}
          </li>
        );
      })}
    </ul>
  );
}
