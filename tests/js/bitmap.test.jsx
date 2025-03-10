import { render, act } from '@testing-library/react';
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Bitmap } from '../../src/genstudio/js/components/bitmap';
import { $StateContext } from '../../src/genstudio/js/context';

// Mock the useContainerWidth hook
vi.mock('../../src/genstudio/js/utils', () => ({
  useContainerWidth: () => [React.createRef(), 500]
}));

// Mock ImageData if not available in test environment
if (typeof ImageData === 'undefined') {
  global.ImageData = class ImageData {
    constructor(data, width, height) {
      this.data = data;
      this.width = width;
      this.height = height;
    }
  };
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('Bitmap Component', () => {
  // Mock canvas and context
  const mockGetContext = vi.fn();
  const mockPutImageData = vi.fn();
  const mockDone = vi.fn();
  const mockBeginUpdate = vi.fn().mockReturnValue(mockDone);

  // Default mock state context
  const defaultStateContext = {
    beginUpdate: mockBeginUpdate
  };

  beforeEach(() => {
    // Reset mocks
    mockGetContext.mockReset();
    mockPutImageData.mockReset();
    mockBeginUpdate.mockReset();
    mockDone.mockReset();
    mockBeginUpdate.mockReturnValue(mockDone);

    // Setup canvas mock
    mockGetContext.mockReturnValue({
      putImageData: mockPutImageData
    });

    // Mock HTMLCanvasElement
    global.HTMLCanvasElement.prototype.getContext = mockGetContext;
  });

  it('should render a canvas element', () => {
    const pixels = new Uint8Array(12); // 4 pixels (RGB)
    const { container } = render(
      <$StateContext.Provider value={defaultStateContext}>
        <Bitmap pixels={pixels} width={2} height={2} />
      </$StateContext.Provider>
    );

    expect(container.querySelector('canvas')).toBeTruthy();
  });

  it('should process RGB pixel data correctly', async () => {
    // Create RGB data (3 bytes per pixel)
    const pixels = new Uint8Array([
      255, 0, 0,    // Red
      0, 255, 0,    // Green
      0, 0, 255,    // Blue
      255, 255, 0   // Yellow
    ]);

    await act(async () => {
      render(
        <$StateContext.Provider value={defaultStateContext}>
          <Bitmap pixels={pixels} width={2} height={2} />
        </$StateContext.Provider>
      );
    });

    // Check if getContext was called with '2d'
    expect(mockGetContext).toHaveBeenCalledWith('2d');

    // Check if putImageData was called
    expect(mockPutImageData).toHaveBeenCalled();

    // Verify the first argument to putImageData is an ImageData
    const imageDataArg = mockPutImageData.mock.calls[0][0];
    expect(imageDataArg).toBeInstanceOf(ImageData);

    // Verify the dimensions
    expect(imageDataArg.width).toBe(2);
    expect(imageDataArg.height).toBe(2);

    // Verify the RGBA data conversion
    const rgba = imageDataArg.data;
    expect(rgba[0]).toBe(255);  // R
    expect(rgba[1]).toBe(0);    // G
    expect(rgba[2]).toBe(0);    // B
    expect(rgba[3]).toBe(255);  // A (added)

    expect(rgba[4]).toBe(0);    // R
    expect(rgba[5]).toBe(255);  // G
    expect(rgba[6]).toBe(0);    // B
    expect(rgba[7]).toBe(255);  // A (added)
  });

  it('should handle RGBA pixel data correctly', async () => {
    // Create RGBA data (4 bytes per pixel)
    const pixels = new Uint8ClampedArray([
      255, 0, 0, 128,     // Red with 50% alpha
      0, 255, 0, 255,     // Green with 100% alpha
      0, 0, 255, 200,     // Blue with ~78% alpha
      255, 255, 0, 100    // Yellow with ~39% alpha
    ]);

    await act(async () => {
      render(
        <$StateContext.Provider value={defaultStateContext}>
          <Bitmap pixels={pixels} width={2} height={2} />
        </$StateContext.Provider>
      );
    });

    // Check if putImageData was called
    expect(mockPutImageData).toHaveBeenCalled();

    // Verify the ImageData
    const imageDataArg = mockPutImageData.mock.calls[0][0];
    expect(imageDataArg).toBeInstanceOf(ImageData);

    // Verify the dimensions
    expect(imageDataArg.width).toBe(2);
    expect(imageDataArg.height).toBe(2);

    // Verify the RGBA data is preserved
    const rgba = imageDataArg.data;
    expect(rgba[0]).toBe(255);  // R
    expect(rgba[1]).toBe(0);    // G
    expect(rgba[2]).toBe(0);    // B
    expect(rgba[3]).toBe(128);  // A (preserved)

    expect(rgba[4]).toBe(0);    // R
    expect(rgba[5]).toBe(255);  // G
    expect(rgba[6]).toBe(0);    // B
    expect(rgba[7]).toBe(255);  // A (preserved)
  });

  it('should apply container width to canvas style', () => {
    const pixels = new Uint8Array(12);
    const { container } = render(
      <$StateContext.Provider value={defaultStateContext}>
        <Bitmap pixels={pixels} width={100} height={50} />
      </$StateContext.Provider>
    );

    const canvas = container.querySelector('canvas');
    expect(canvas).toBeTruthy();
    expect(canvas.style.width).toBe('500px'); // From mocked useContainerWidth
    expect(canvas.width).toBe(100);
    expect(canvas.height).toBe(50);
  });

  it('should call beginUpdate and done from state context', async () => {
    // Create a spy for the done function
    const doneSpy = vi.fn();
    const beginUpdateSpy = vi.fn().mockReturnValue(doneSpy);

    // Create a mock state context
    const mockStateContext = {
      beginUpdate: beginUpdateSpy
    };

    // Render with mocked context provider
    const pixels = new Uint8Array(12);

    await act(async () => {
      render(
        <$StateContext.Provider value={mockStateContext}>
          <Bitmap pixels={pixels} width={2} height={2} />
        </$StateContext.Provider>
      );
    });

    // Verify beginUpdate was called with "bitmap"
    expect(beginUpdateSpy).toHaveBeenCalledWith("bitmap");

    // Verify done was called after processing
    expect(doneSpy).toHaveBeenCalled();
  });

  it('should not crash if canvas context is null', async () => {
    // Mock getContext to return null
    mockGetContext.mockReturnValueOnce(null);

    const pixels = new Uint8Array(12);

    await act(async () => {
      // This should not throw
      expect(() => {
        render(
          <$StateContext.Provider value={defaultStateContext}>
            <Bitmap pixels={pixels} width={2} height={2} />
          </$StateContext.Provider>
        );
      }).not.toThrow();
    });

    // putImageData should not have been called
    expect(mockPutImageData).not.toHaveBeenCalled();
  });

  it('should handle empty pixel data gracefully', async () => {
    const pixels = new Uint8Array(0);

    await act(async () => {
      // This should not throw
      expect(() => {
        render(
          <$StateContext.Provider value={defaultStateContext}>
            <Bitmap pixels={pixels} width={0} height={0} />
          </$StateContext.Provider>
        );
      }).not.toThrow();
    });
  });
});
