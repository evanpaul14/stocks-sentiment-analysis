// Re-mounts on every navigation (unlike layout.tsx), so each page gets a
// fresh fade/slide-in instead of snapping in instantly.
export default function Template({ children }: { children: React.ReactNode }) {
  return <div className="animate-page-in">{children}</div>;
}
