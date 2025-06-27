/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const isProd = process.env.NODE_ENV === 'production';

    return [
      {
        source: '/api/llm/:path*',
        destination: isProd
          ? 'http://31.220.104.116:8000/llm/:path*'
          : 'http://127.0.0.1:8000/llm/:path*',
      },
      {
        source: '/api/agent',
        destination: isProd
          ? 'http://31.220.104.116:8000/agent'
          : 'http://127.0.0.1:8000/agent',
      },
      {
        source: '/api/search',
        destination: isProd
          ? 'http://31.220.104.116:8000/search'
          : 'http://127.0.0.1:8000/search',
      },
    ];
  },

  httpAgentOptions: {
    keepAlive: true,
  },

  experimental: {
    proxyTimeout: 120_000, // 120 segundos
  },
};

export default nextConfig;
