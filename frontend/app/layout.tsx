import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "AegisKEV | SOC Analyst Workbench",
  description:
    "Point-in-time 180-day CISA KEV exploitation forecasting with conformal uncertainty and TreeSHAP explanations.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-transparent text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
