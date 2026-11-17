#pragma once

#include <vector>

// Serial reference implementation.
// Uses clamp-to-edge boundary handling.
void boxFilterSerial(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int width,
    int height,
    int radius
);

// Strip decomposition.
// Input contains top/bottom halo rows.
// Horizontal boundaries are handled with clamp-to-edge.
void boxFilterLocalStrip(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int width,
    int localHeight,
    int radius
);

// Complete 2D block filter.
// Input contains halo rows and halo columns.
void boxFilterLocalBlock(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int localWidth,
    int localHeight,
    int radius
);

// Compute only a selected region of a 2D block.
// Coordinates are in local output coordinates.
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
);