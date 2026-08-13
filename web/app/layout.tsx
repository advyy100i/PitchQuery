import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Newsreader } from "next/font/google";
// @ts-expect-error - Next.js handles CSS imports at build time.
import "./globals.css";

/**
 * Three families, each doing one job:
 *   Newsreader   — the wordmark and section headings. A serif gives the page an
 *                  editorial register rather than a dashboard one.
 *   IBM Plex Sans — interface text. Engineered rather than neutral, which suits
 *                  a tool about measurement, and avoids the Inter/Roboto default.
 *   IBM Plex Mono — token strings and IDs, where character alignment is
 *                  meaningful and the grammar should read as notation.
 *
 * next/font self-hosts these at build time, so there is no external request,
 * no layout shift, and the app still works offline.
 */

const sans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

const serif = Newsreader({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  style: ["normal", "italic"],
  variable: "--font-serif",
  display: "swap",
});

export const metadata: Metadata = {
  title: "PitchQuery — tactical possession search",
  description:
    "Search 66,817 football possessions by tactical pattern. Built on StatsBomb open data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  );
}
