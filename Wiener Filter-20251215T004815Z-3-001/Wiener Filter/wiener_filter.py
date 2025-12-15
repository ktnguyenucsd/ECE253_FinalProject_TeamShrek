import argparse
import cv2
import numpy as np

def wiener_filter_channel(blur_image, kernel, lamda):
    kernel_pad = np.zeros(blur_image.shape)
    kh, kw = kernel.shape
    kernel_pad[:kh, :kw] = kernel
    kernel_pad = np.roll(kernel_pad, -kh // 2, axis=0)
    kernel_pad = np.roll(kernel_pad, -kw // 2, axis=1)
    blur_image_fft = np.fft.fft2(blur_image)
    kernel_fft = np.fft.fft2(kernel_pad)

    kernel_fft_conj = np.conj(kernel_fft)
    denominator = np.abs(kernel_fft) ** 2 + lamda
    clean_image_fft = blur_image_fft * (kernel_fft_conj / denominator)
    clean_image = np.fft.ifft2(clean_image_fft)
    clean_image = np.real(clean_image)
    clean_image = np.clip(clean_image, 0, 255).astype(np.uint8)
    return clean_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str)
    parser.add_argument('direction', type=int)
    parser.add_argument('length', type=int)
    parser.add_argument('lamda', type=float)
    args = parser.parse_args()
    img = cv2.imread(args.input_image)
    length = args.length
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length))
    center = length // 2
    rad = np.deg2rad(args.direction)
    dx = np.cos(rad)
    dy = np.sin(rad)
    for i in range(length):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    kernel /= np.sum(kernel)
    clean = np.zeros_like(img)
    for c in range(3):
        clean[:, :, c] = wiener_filter_channel(img[:, :, c], kernel, args.lamda)
    output_path = args.input_image.rsplit('.', 1)[0] + '_clean.png'
    cv2.imwrite(output_path, clean)

if __name__ == "__main__":
    main()