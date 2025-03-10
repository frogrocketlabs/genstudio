import * as React from 'react';
import { useContext, useEffect, useMemo, useRef } from 'react';
import { $StateContext } from '../context';
import { useContainerWidth } from '../utils';

/**
 * Props interface for the bitmap component
 */
interface BitmapProps {
    /** Raw pixel data as Uint8Array/Uint8ClampedArray */
    pixels: Uint8Array | Uint8ClampedArray;
    /** Width of the bitmap in pixels */
    width: number;
    /** Height of the bitmap in pixels */
    height: number;
    /** CSS styles to apply to the canvas element */
    style?: React.CSSProperties;
    /** How to interpolate pixels when scaling */
    interpolation?: 'nearest' | 'bilinear';
}

/**
 * Renders raw pixel data as a bitmap image in a canvas element.
 *
 * Supports both RGB (3 bytes per pixel) and RGBA (4 bytes per pixel) formats.
 * RGB data is automatically converted to RGBA by setting alpha to 255.
 * The canvas is scaled to fit the container width while maintaining aspect ratio.
 *
 * @param pixels - Raw pixel data as Uint8Array/Uint8ClampedArray
 * @param width - Width of the bitmap in pixels
 * @param height - Height of the bitmap in pixels
 * @param interpolation - How to interpolate pixels when scaling
 * @returns React component rendering the bitmap
 */
export function Bitmap({pixels, width, height, interpolation = 'nearest', style, ...props}: BitmapProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const [ref, containerWidth] = useContainerWidth();
    const $state = useContext($StateContext);
    const done = useMemo(() => $state.beginUpdate("bitmap"), [])

    useEffect(() => {

        const canvas = canvasRef.current;
        if (!canvas) {
            console.warn('Canvas ref not available');
            return;
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) {
            console.warn('Could not get 2d context');
            return;
        }

        const bytesPerPixel = pixels.length / (width * height);

        // Create ImageData based on pixel format
        let imageData: ImageData;
        if (bytesPerPixel === 3) {
            // Convert RGB to RGBA
            const rgba = new Uint8ClampedArray(width * height * 4);
            for (let i = 0; i < pixels.length; i += 3) {
                const j = (i / 3) * 4;
                rgba[j] = pixels[i];
                rgba[j + 1] = pixels[i + 1];
                rgba[j + 2] = pixels[i + 2];
                rgba[j + 3] = 255;
            }
            imageData = new ImageData(rgba, width, height);
        } else {
            // Assume RGBA
            imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
        }

        ctx.putImageData(imageData, 0, 0);
        done();
    }, [pixels, width, height, interpolation]);

    return <div ref={ref}>
        <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{
                width: containerWidth,
                imageRendering: interpolation === 'nearest' ? 'pixelated' : 'auto',
                ...style,
            }}
            {...props}
        />
    </div>;
}
