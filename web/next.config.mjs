/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    // The FastAPI layer. Override with NEXT_PUBLIC_API_URL when deploying.
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
};

export default nextConfig;
