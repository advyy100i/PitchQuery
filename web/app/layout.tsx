import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

/**
 * Three families, each doing one job:
 *   Newsreader    — the wordmark, headings and the scouting note. A serif gives
 *                   the page an editorial register rather than a dashboard one.
 *   IBM Plex Sans — interface text. Engineered rather than neutral, which suits
 *                   a tool about measurement, and avoids the Inter/Roboto default.
 *   IBM Plex Mono — token strings and IDs, where character alignment is
 *                   meaningful and the grammar should read as notation.
 *
 * These are SELF-HOSTED from app/fonts rather than pulled with
 * next/font/google. The Google loader fetches from fonts.gstatic.com at BUILD
 * time, which makes every production build a network call — and Vercel's build
 * container could not reach it, so the deploy failed outright. Shipping the
 * files makes the build hermetic: it works offline and cannot break because a
 * font CDN is unreachable from wherever it happens to run.
 *
 * Latin subset only, at the weights actually used: 337 KB for all eight files.
 */

const sans = localFont({
  src: [
    { path: "./fonts/IBMPlexSans-Regular.woff2", weight: "400", style: "normal" },
    { path: "./fonts/IBMPlexSans-Medium.woff2", weight: "500", style: "normal" },
    { path: "./fonts/IBMPlexSans-SemiBold.woff2", weight: "600", style: "normal" },
  ],
  variable: "--font-sans",
  display: "swap",
});

const mono = localFont({
  src: [{ path: "./fonts/IBMPlexMono-Regular.woff2", weight: "400", style: "normal" }],
  variable: "--font-mono",
  display: "swap",
});

const serif = localFont({
  src: [
    { path: "./fonts/Newsreader-Regular.woff2", weight: "400", style: "normal" },
    { path: "./fonts/Newsreader-RegularItalic.woff2", weight: "400", style: "italic" },
    { path: "./fonts/Newsreader-Medium.woff2", weight: "500", style: "normal" },
    { path: "./fonts/Newsreader-SemiBold.woff2", weight: "600", style: "normal" },
  ],
  variable: "--font-serif",
  display: "swap",
});

export const metadata: Metadata = {
  title: "PitchQuery — tactical possession search",
  description:
    "Search football possessions by tactical pattern. Built on StatsBomb open data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  );
}
