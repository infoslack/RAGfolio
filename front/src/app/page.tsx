'use client';

import { Container } from "@/components/container";
import { StreamingChat } from "@/components/streaming-chat";
import { UploadModal } from "@/components/upload-modal";
import { useState } from "react";
import { LineChart } from "lucide-react";

export default function Home() {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  return (
    <div className="flex flex-col min-h-screen">
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur">
        <Container className="flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <LineChart className="h-6 w-6 text-primary" />
            <span className="font-semibold text-foreground text-xl">RAGfolio</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsUploadModalOpen(true)}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Add
            </button>
          </div>
        </Container>
      </header>

      <main className="flex-1 py-8">
        <Container>
          <div className="mx-auto max-w-4xl">
            <h1 className="text-3xl font-bold mb-6 text-foreground">Investment Intelligence at Your Fingertips</h1>
            <p className="text-muted-foreground mb-8">
              Ask anything about companies, stocks, or market trends. Enable Multi-Stream Analysis for comprehensive investment insights powered by fundamental, momentum, and sentiment analysis.
            </p>
            <StreamingChat />
          </div>
        </Container>
      </main>

      <footer className="border-t border-border py-6 bg-muted">
        <Container className="flex items-center justify-start">
          <p className="text-sm text-muted-foreground">
            © {new Date().getFullYear()} TechLevel Pro
          </p>
        </Container>
      </footer>

      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
      />
    </div>
  );
}