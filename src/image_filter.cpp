#include "image_filter.h"

#include <algorithm>

static inline int clampInt(int value, int low, int high)
{
    return std::max(low, std::min(value, high));
}


// ============================================================
// SERIAL REFERENCE
// ============================================================

void boxFilterSerial(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int width,
    int height,
    int radius
)
{
    output.resize(width * height);

    const int kernelSize = 2 * radius + 1;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            long long sum = 0;

            for (int dy = -radius; dy <= radius; ++dy)
            {
                int ny = clampInt(y + dy, 0, height - 1);

                for (int dx = -radius; dx <= radius; ++dx)
                {
                    int nx = clampInt(x + dx, 0, width - 1);

                    sum += input[ny * width + nx];
                }
            }

            output[y * width + x] =
                static_cast<unsigned char>(
                    sum / (kernelSize * kernelSize)
                );
        }
    }
}


// ============================================================
// STRIP DECOMPOSITION
// ============================================================

void boxFilterLocalStrip(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int width,
    int localHeight,
    int radius
)
{
    output.resize(width * localHeight);

    const int kernelSize = 2 * radius + 1;

    // Input layout:
    //
    //     top halo
    //     actual local rows
    //     bottom halo
    //
    const int extendedHeight = localHeight + 2 * radius;

    for (int y = 0; y < localHeight; ++y)
    {
        int centerY = y + radius;

        for (int x = 0; x < width; ++x)
        {
            long long sum = 0;

            for (int dy = -radius; dy <= radius; ++dy)
            {
                int ny = centerY + dy;

                for (int dx = -radius; dx <= radius; ++dx)
                {
                    int nx = clampInt(
                        x + dx,
                        0,
                        width - 1
                    );

                    sum += input[ny * width + nx];
                }
            }

            output[y * width + x] =
                static_cast<unsigned char>(
                    sum / (kernelSize * kernelSize)
                );
        }
    }
}


// ============================================================
// 2D BLOCK - FULL IMAGE
// ============================================================

void boxFilterLocalBlock(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int localWidth,
    int localHeight,
    int radius
)
{
    output.resize(localWidth * localHeight);

    boxFilterLocalBlockRegion(
        input,
        output,
        localWidth,
        localHeight,
        radius,
        0,
        localHeight,
        0,
        localWidth
    );
}


// ============================================================
// 2D BLOCK - SELECTED REGION
// ============================================================

void boxFilterLocalBlockRegion(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int localWidth,
    int localHeight,
    int radius,
    int yStart,
    int yEnd,
    int xStart,
    int xEnd
)
{
    if (output.size() !=
        static_cast<size_t>(localWidth * localHeight))
    {
        output.resize(localWidth * localHeight);
    }

    const int kernelSize = 2 * radius + 1;

    // Extended local image:
    //
    //       halo
    //   +-----------+
    //   |           |
    //   |   local   |
    //   |   block   |
    //   |           |
    //   +-----------+
    //
    // Width = localWidth + 2*radius

    const int extendedWidth =
        localWidth + 2 * radius;

    for (int y = yStart; y < yEnd; ++y)
    {
        for (int x = xStart; x < xEnd; ++x)
        {
            long long sum = 0;

            int centerY = y + radius;
            int centerX = x + radius;

            for (int dy = -radius; dy <= radius; ++dy)
            {
                int ny = centerY + dy;

                for (int dx = -radius; dx <= radius; ++dx)
                {
                    int nx = centerX + dx;

                    sum += input[
                        ny * extendedWidth + nx
                    ];
                }
            }

            output[y * localWidth + x] =
                static_cast<unsigned char>(
                    sum / (kernelSize * kernelSize)
                );
        }
    }
}