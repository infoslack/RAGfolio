/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const isProd = process.env.NODE_ENV === 'production';
    const isDocker = process.env.DOCKER_ENV === 'true';

    const backendUrl = isProd
      ? 'http://31.220.104.116:8000'
      : isDocker
        ? 'http://backend:8000'
        : 'http://127.0.0.1:8000';

    return [
      {
        source: '/api/llm/:path*',
        destination: `${backendUrl}/llm/:path*`,
      },
      {
        source: '/api/agent',
        destination: `${backendUrl}/agent`,
      },
      {
        source: '/api/search',
        destination: `${backendUrl}/search`,
      },
    ];
  },

  httpAgentOptions: {
    keepAlive: true,
  },

  experimental: {
    proxyTimeout: 120_000, // 120 seconds
  },
};

export default nextConfig;