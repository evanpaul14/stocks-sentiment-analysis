"use client";

import { useEffect, useRef } from "react";

interface Particle {
  x: number;
  y: number;
  px: number;
  py: number;
  vx: number;
  vy: number;
  life: number;
}

const PARTICLE_COUNT = 160;

/** Cheap pseudo-noise flow field (sin/cos composition) — avoids an extra
 * dependency for a visual effect that doesn't need true simplex noise. */
function flowAngle(x: number, y: number, t: number): number {
  return (
    Math.sin(x * 0.0025 + t * 0.15) * Math.PI +
    Math.cos(y * 0.0025 - t * 0.1) * Math.PI
  );
}

export function ParticleFlowField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -9999, y: -9999 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    if (prefersReducedMotion) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Canvas 2D fillStyle/strokeStyle can't resolve CSS custom properties
    // (there's no cascade context for a canvas draw call) — passing
    // "var(--color-chart-1)" directly silently no-ops and leaves the
    // default black, invisible against this app's dark theme. Resolve the
    // actual computed colors once via a throwaway element (handles the
    // --color-chart-1 -> --chart-1 alias chain too).
    function resolveColor(varExpr: string): string {
      const probe = document.createElement("span");
      probe.style.color = varExpr;
      probe.style.display = "none";
      document.body.appendChild(probe);
      const resolved = getComputedStyle(probe).color;
      probe.remove();
      return resolved;
    }
    const bullColor = resolveColor("var(--color-chart-1)");
    const bearColor = resolveColor("var(--color-chart-3)");

    let width = 0;
    let height = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    function resize() {
      const canvas = canvasRef.current;
      if (!canvas || !ctx) return;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    }
    resize();
    window.addEventListener("resize", resize);

    function handlePointerMove(e: PointerEvent) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }
    window.addEventListener("pointermove", handlePointerMove);

    const particles: Particle[] = Array.from({ length: PARTICLE_COUNT }, () => {
      const x = Math.random() * width;
      const y = Math.random() * height;
      return { x, y, px: x, py: y, vx: 0, vy: 0, life: Math.random() * 200 };
    });

    let animationFrame: number | null = null;
    let t = 0;
    // Only animate while the canvas is scrolled into view and the tab is
    // active — this runs forever otherwise, burning CPU/GPU for a purely
    // decorative background nobody is looking at.
    let isVisible = true;
    let isPageVisible = document.visibilityState === "visible";

    function step() {
      if (!ctx) return;
      t += 1;
      ctx.clearRect(0, 0, width, height);

      for (const p of particles) {
        p.px = p.x;
        p.py = p.y;

        const angle = flowAngle(p.x, p.y, t * 0.01);
        p.vx += Math.cos(angle) * 0.05;
        p.vy += Math.sin(angle) * 0.05;

        const dx = p.x - mouseRef.current.x;
        const dy = p.y - mouseRef.current.y;
        const distSq = dx * dx + dy * dy;
        if (distSq < 10000) {
          const dist = Math.sqrt(distSq) || 1;
          p.vx += (dx / dist) * 0.6;
          p.vy += (dy / dist) * 0.6;
        }

        p.vx *= 0.94;
        p.vy *= 0.94;
        p.x += p.vx;
        p.y += p.vy;
        p.life -= 1;

        if (p.life <= 0 || p.x < 0 || p.x > width || p.y < 0 || p.y > height) {
          p.x = Math.random() * width;
          p.y = Math.random() * height;
          p.px = p.x;
          p.py = p.y;
          p.vx = 0;
          p.vy = 0;
          p.life = 150 + Math.random() * 150;
          continue;
        }

        // Bullish (moving up the screen) reads green, bearish reads red —
        // ticks read as a scattered price feed rather than generic dust.
        const bullish = p.vy < 0;
        const speed = Math.min(Math.hypot(p.vx, p.vy), 3);
        const color = bullish ? bullColor : bearColor;

        ctx.beginPath();
        ctx.moveTo(p.px, p.py);
        ctx.lineTo(p.x, p.y);
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.55 + speed * 0.12;
        ctx.lineWidth = 1.5;
        ctx.lineCap = "round";
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.9;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      animationFrame = requestAnimationFrame(step);
    }

    function start() {
      if (animationFrame !== null) return;
      animationFrame = requestAnimationFrame(step);
    }
    function stop() {
      if (animationFrame === null) return;
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
    function syncRunning() {
      if (isVisible && isPageVisible) {
        start();
      } else {
        stop();
      }
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        isVisible = entry.isIntersecting;
        syncRunning();
      },
      { threshold: 0 }
    );
    observer.observe(canvas);

    function handleVisibilityChange() {
      isPageVisible = document.visibilityState === "visible";
      syncRunning();
    }
    document.addEventListener("visibilitychange", handleVisibilityChange);

    syncRunning();

    return () => {
      stop();
      observer.disconnect();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", handlePointerMove);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 h-full w-full opacity-60"
    />
  );
}
