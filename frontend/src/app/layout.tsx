import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "Voice Support Agent",
  description: "Live customer-support voice demo",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
