/**
 * Ready state tracking system for GenStudio components
 *
 * This module provides utilities for tracking readiness of asynchronous components
 * like WebGPU rendering in Scene3d. Components can register themselves as "loading"
 * and signal when they're "ready", and other parts of the application can wait for
 * all components to be ready.
 */

import * as React from 'react';

const DEBUG = false;

const log = (...body: any[]) => {
  if (!DEBUG) return;
  console.log(...body)
}

/**
 * Global ready state manager that tracks pending async operations
 */
class ReadyStateManager {
  private pendingCount = 0;
  private readyPromise: Promise<void> | null = null;
  private resolveReady: (() => void) | null = null;

  /**
   * Increment the pending counter, indicating an async operation has started
   * @returns A function to call when the operation completes
   */
  public beginUpdate(label: string): () => void {
    let valid = true;
    this.pendingCount++;
    log(`[ReadyState] Begin ${label}, pending count: ${this.pendingCount}`);
    this.ensurePromise();

    return () => {
      if (!valid) return;
      valid = false;
      this.pendingCount--;
      log(`[ReadyState] End ${label}, pending count: ${this.pendingCount}`);
      if (this.pendingCount === 0 && this.resolveReady) {
        log("[ReadyState] All updates complete, resolving ready promise");
        this.resolveReady();
        this.readyPromise = null;
        this.resolveReady = null;
      }
    };
  }

  /**
   * Returns a promise that resolves when all pending operations are complete
   */
  public async whenReady(): Promise<void> {
    if (this.pendingCount === 0) {
      return Promise.resolve();
    }

    log(`[ReadyState] whenReady called, waiting for ${this.pendingCount} pending updates`);
    this.ensurePromise();
    return this.readyPromise!;
  }

  /**
   * Returns true if there are no pending operations
   */
  public isReady(): boolean {
    return this.pendingCount === 0;
  }

  /**
   * Reset the ready state for testing purposes
   */
  public reset(): void {
    this.pendingCount = 0;
    this.readyPromise = null;
    this.resolveReady = null;
  }

  private ensurePromise(): void {
    if (!this.readyPromise) {
      this.readyPromise = new Promise<void>((resolve) => {
        this.resolveReady = resolve;
      });
    }
  }
}

// Export a singleton instance for the application
export const readyState = new ReadyStateManager();

/**
 * Hook to track and signal readiness for React components
 *
 * @param isLoading Whether the component is currently loading
 * @returns Object with signal completion function
 */
export function useReadySignal(label: string, isLoading: boolean): void {
  const completeRef = React.useRef<(() => void) | null>(null);

  React.useEffect(() => {
    // Always clean up previous signal if any
    if (completeRef.current) {
      completeRef.current();
      completeRef.current = null;
    }

    // If component is loading, register a new loading state
    if (isLoading) {
      completeRef.current = readyState.beginUpdate(label);
    }

    // Clean up on unmount
    return () => {
      if (completeRef.current) {
        completeRef.current();
        completeRef.current = null;
      }
    };
  }, [isLoading]);

}

// Provide access to whenReady for screenshots
if (typeof window !== 'undefined') {
  (window as any).genStudioReadyState = readyState;
}
