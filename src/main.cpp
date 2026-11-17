#include <mpi.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <direct.h>

#include "image_filter.h"

using namespace std;
using namespace cv;


// ============================================================
// CREATE OUTPUT DIRECTORY
// ============================================================

void createOutputDirectory()
{
    _mkdir("output");
}


// ============================================================
// STRIP HALO EXCHANGE
// ============================================================

void exchangeStripHalos(
    vector<unsigned char>& extended,
    int width,
    int localHeight,
    int radius,
    int rank,
    int worldSize
)
{
    int topRank =
        (rank == 0) ? MPI_PROC_NULL : rank - 1;

    int bottomRank =
        (rank == worldSize - 1)
            ? MPI_PROC_NULL
            : rank + 1;

    int haloBytes =
        radius * width;

    unsigned char* topHalo =
        extended.data();

    unsigned char* firstActualRow =
        extended.data() +
        radius * width;

    unsigned char* lastActualRows =
        extended.data() +
        radius * width +
        (localHeight - radius) * width;

    unsigned char* bottomHalo =
        extended.data() +
        (radius + localHeight) * width;

    MPI_Request requests[4];

    MPI_Irecv(
        topHalo,
        haloBytes,
        MPI_UNSIGNED_CHAR,
        topRank,
        200,
        MPI_COMM_WORLD,
        &requests[0]
    );

    MPI_Irecv(
        bottomHalo,
        haloBytes,
        MPI_UNSIGNED_CHAR,
        bottomRank,
        201,
        MPI_COMM_WORLD,
        &requests[1]
    );

    MPI_Isend(
        firstActualRow,
        haloBytes,
        MPI_UNSIGNED_CHAR,
        topRank,
        201,
        MPI_COMM_WORLD,
        &requests[2]
    );

    MPI_Isend(
        lastActualRows,
        haloBytes,
        MPI_UNSIGNED_CHAR,
        bottomRank,
        200,
        MPI_COMM_WORLD,
        &requests[3]
    );

    MPI_Waitall(
        4,
        requests,
        MPI_STATUSES_IGNORE
    );

    // Global top boundary
    if (rank == 0)
    {
        for (int r = 0; r < radius; ++r)
        {
            memcpy(
                extended.data() + r * width,
                extended.data() + radius * width,
                width
            );
        }
    }

    // Global bottom boundary
    if (rank == worldSize - 1)
    {
        int lastActualRow =
            radius + localHeight - 1;

        for (int r = 0; r < radius; ++r)
        {
            memcpy(
                extended.data() +
                    (radius + localHeight + r) * width,

                extended.data() +
                    lastActualRow * width,

                width
            );
        }
    }
}


// ============================================================
// 2D BLOCK HALO EXCHANGE
// ============================================================

void exchangeBlockHalos(
    vector<unsigned char>& extended,
    int localWidth,
    int localHeight,
    int radius,
    MPI_Comm cartComm,
    int leftRank,
    int rightRank,
    int topRank,
    int bottomRank
)
{
    int extendedWidth =
        localWidth + 2 * radius;

    int extendedHeight =
        localHeight + 2 * radius;

    // --------------------------------------------------------
    // COLUMN DATATYPE
    // --------------------------------------------------------

    MPI_Datatype columnType;

    MPI_Type_vector(
        extendedHeight,
        radius,
        extendedWidth,
        MPI_UNSIGNED_CHAR,
        &columnType
    );

    MPI_Type_commit(
        &columnType
    );

    // --------------------------------------------------------
    // LEFT / RIGHT HALOS
    // --------------------------------------------------------

    MPI_Request requests[4];

    // Receive left halo
    MPI_Irecv(
        extended.data(),
        1,
        columnType,
        leftRank,
        300,
        cartComm,
        &requests[0]
    );

    // Receive right halo
    MPI_Irecv(
        extended.data() +
            radius +
            localWidth,

        1,
        columnType,

        rightRank,
        301,
        cartComm,
        &requests[1]
    );

    // Send left actual columns
    MPI_Isend(
        extended.data() +
            radius,

        1,
        columnType,

        leftRank,
        301,
        cartComm,
        &requests[2]
    );

    // Send right actual columns
    MPI_Isend(
        extended.data() +
            localWidth,

        1,
        columnType,

        rightRank,
        300,
        cartComm,
        &requests[3]
    );

    MPI_Waitall(
        4,
        requests,
        MPI_STATUSES_IGNORE
    );

    // --------------------------------------------------------
    // TOP / BOTTOM HALOS
    // --------------------------------------------------------

    int rowWidth =
        extendedWidth;

    unsigned char* topHalo =
        extended.data();

    unsigned char* firstActualRows =
        extended.data() +
        radius * extendedWidth;

    unsigned char* bottomHalo =
        extended.data() +
        (radius + localHeight) *
        extendedWidth;

    unsigned char* lastActualRows =
        extended.data() +
        localHeight *
        extendedWidth;

    MPI_Irecv(
        topHalo,
        radius * rowWidth,
        MPI_UNSIGNED_CHAR,
        topRank,
        400,
        cartComm,
        &requests[0]
    );

    MPI_Irecv(
        bottomHalo,
        radius * rowWidth,
        MPI_UNSIGNED_CHAR,
        bottomRank,
        401,
        cartComm,
        &requests[1]
    );

    MPI_Isend(
        firstActualRows,
        radius * rowWidth,
        MPI_UNSIGNED_CHAR,
        topRank,
        401,
        cartComm,
        &requests[2]
    );

    MPI_Isend(
        lastActualRows,
        radius * rowWidth,
        MPI_UNSIGNED_CHAR,
        bottomRank,
        400,
        cartComm,
        &requests[3]
    );

    MPI_Waitall(
        4,
        requests,
        MPI_STATUSES_IGNORE
    );

    MPI_Type_free(
        &columnType
    );

    // --------------------------------------------------------
    // GLOBAL TOP BOUNDARY
    // --------------------------------------------------------

    if (topRank == MPI_PROC_NULL)
    {
        unsigned char* firstActual =
            extended.data() +
            radius * extendedWidth;

        for (int r = 0; r < radius; ++r)
        {
            memcpy(
                extended.data() +
                    r * extendedWidth,

                firstActual,

                extendedWidth
            );
        }
    }

    // --------------------------------------------------------
    // GLOBAL BOTTOM BOUNDARY
    // --------------------------------------------------------

    if (bottomRank == MPI_PROC_NULL)
    {
        unsigned char* lastActual =
            extended.data() +
            (radius + localHeight - 1) *
            extendedWidth;

        for (int r = 0; r < radius; ++r)
        {
            memcpy(
                extended.data() +
                    (radius + localHeight + r) *
                    extendedWidth,

                lastActual,

                extendedWidth
            );
        }
    }

    // --------------------------------------------------------
    // GLOBAL LEFT BOUNDARY
    // --------------------------------------------------------

    if (leftRank == MPI_PROC_NULL)
    {
        for (int y = 0;
             y < extendedHeight;
             ++y)
        {
            unsigned char value =
                extended[
                    y * extendedWidth +
                    radius
                ];

            for (int x = 0;
                 x < radius;
                 ++x)
            {
                extended[
                    y * extendedWidth + x
                ] = value;
            }
        }
    }

    // --------------------------------------------------------
    // GLOBAL RIGHT BOUNDARY
    // --------------------------------------------------------

    if (rightRank == MPI_PROC_NULL)
    {
        for (int y = 0;
             y < extendedHeight;
             ++y)
        {
            unsigned char value =
                extended[
                    y * extendedWidth +
                    radius +
                    localWidth - 1
                ];

            for (int x = 0;
                 x < radius;
                 ++x)
            {
                extended[
                    y * extendedWidth +
                    radius +
                    localWidth +
                    x
                ] = value;
            }
        }
    }
}


// ============================================================
// CALCULATE 2D BLOCK SIZE
// ============================================================

void calculateBlockSize(
    int globalWidth,
    int globalHeight,
    int dims[2],
    int coords[2],
    int& localWidth,
    int& localHeight,
    int& startX,
    int& startY
)
{
    int baseHeight =
        globalHeight / dims[0];

    int remainderHeight =
        globalHeight % dims[0];

    localHeight =
        baseHeight +
        (coords[0] < remainderHeight ? 1 : 0);

    startY =
        coords[0] * baseHeight +
        min(
            coords[0],
            remainderHeight
        );

    int baseWidth =
        globalWidth / dims[1];

    int remainderWidth =
        globalWidth % dims[1];

    localWidth =
        baseWidth +
        (coords[1] < remainderWidth ? 1 : 0);

    startX =
        coords[1] * baseWidth +
        min(
            coords[1],
            remainderWidth
        );
}


// ============================================================
// PACK 2D BLOCKS
// ============================================================

void pack2DBlocks(
    const vector<unsigned char>& image,
    vector<unsigned char>& packed,
    int width,
    int height,
    int dims[2]
)
{
    int worldSize =
        dims[0] * dims[1];

    int totalSize = 0;

    for (int r = 0;
         r < worldSize;
         ++r)
    {
        int coords[2];

        coords[0] =
            r / dims[1];

        coords[1] =
            r % dims[1];

        int localWidth;
        int localHeight;
        int startX;
        int startY;

        calculateBlockSize(
            width,
            height,
            dims,
            coords,
            localWidth,
            localHeight,
            startX,
            startY
        );

        totalSize +=
            localWidth *
            localHeight;
    }

    packed.resize(
        totalSize
    );

    int offset = 0;

    for (int r = 0;
         r < worldSize;
         ++r)
    {
        int coords[2];

        coords[0] =
            r / dims[1];

        coords[1] =
            r % dims[1];

        int localWidth;
        int localHeight;
        int startX;
        int startY;

        calculateBlockSize(
            width,
            height,
            dims,
            coords,
            localWidth,
            localHeight,
            startX,
            startY
        );

        for (int y = 0;
             y < localHeight;
             ++y)
        {
            memcpy(
                packed.data() +
                    offset +
                    y * localWidth,

                image.data() +
                    (startY + y) * width +
                    startX,

                localWidth
            );
        }

        offset +=
            localWidth *
            localHeight;
    }
}


// ============================================================
// UNPACK 2D BLOCKS
// ============================================================

void unpack2DBlocks(
    const vector<unsigned char>& packed,
    vector<unsigned char>& image,
    int width,
    int height,
    int dims[2]
)
{
    int worldSize =
        dims[0] * dims[1];

    int offset = 0;

    for (int r = 0;
         r < worldSize;
         ++r)
    {
        int coords[2];

        coords[0] =
            r / dims[1];

        coords[1] =
            r % dims[1];

        int localWidth;
        int localHeight;
        int startX;
        int startY;

        calculateBlockSize(
            width,
            height,
            dims,
            coords,
            localWidth,
            localHeight,
            startX,
            startY
        );

        for (int y = 0;
             y < localHeight;
             ++y)
        {
            memcpy(
                image.data() +
                    (startY + y) * width +
                    startX,

                packed.data() +
                    offset +
                    y * localWidth,

                localWidth
            );
        }

        offset +=
            localWidth *
            localHeight;
    }
}


// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv)
{
    MPI_Init(
        &argc,
        &argv
    );

    int rank;
    int worldSize;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &worldSize
    );

    string mode = "strip";
    int radius = 1;

    // --------------------------------------------------------
    // COMMAND LINE
    // --------------------------------------------------------

    for (int i = 1;
         i < argc;
         ++i)
    {
        string arg =
            argv[i];

        if (arg == "--mode" &&
            i + 1 < argc)
        {
            mode =
                argv[++i];
        }
        else if (
            arg == "--radius" &&
            i + 1 < argc)
        {
            radius =
                atoi(argv[++i]);
        }
    }

    if (radius < 1)
    {
        if (rank == 0)
        {
            cerr
                << "ERROR: Radius must be >= 1."
                << endl;
        }

        MPI_Finalize();

        return 1;
    }

    // ========================================================
    // INPUT
    // ========================================================

    double inputStart =
        MPI_Wtime();

    vector<unsigned char> image;

    int width = 0;
    int height = 0;

    if (rank == 0)
    {
        Mat inputImage =
            imread(
                "images/input.jpg",
                IMREAD_GRAYSCALE
            );

        if (inputImage.empty())
        {
            cerr
                << "ERROR: Cannot load "
                   "images/input.jpg"
                << endl;

            MPI_Abort(
                MPI_COMM_WORLD,
                1
            );
        }

        width =
            inputImage.cols;

        height =
            inputImage.rows;

        image.resize(
            width * height
        );

        memcpy(
            image.data(),
            inputImage.data,
            width * height
        );
    }

    MPI_Bcast(
        &width,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        &height,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    double inputEnd =
        MPI_Wtime();

    // ========================================================
    // HEADER
    // ========================================================

    if (rank == 0)
    {
        cout << endl;
        cout << "====================================="
             << endl;
        cout << "MPI Image Filter"
             << endl;
        cout << "====================================="
             << endl;

        cout << "Image: "
             << width
             << " x "
             << height
             << endl;

        cout << "MPI ranks: "
             << worldSize
             << endl;

        cout << "Mode: "
             << mode
             << endl;

        cout << "Radius: "
             << radius
             << endl;

        cout << endl;
    }


    // ========================================================
    // STRIP MODE
    // ========================================================

    if (mode == "strip")
    {
        if (rank == 0)
        {
            cout
                << "-------------------------------------"
                << endl;

            cout
                << "STRIP DECOMPOSITION"
                << endl;

            cout
                << "-------------------------------------"
                << endl;
        }

        int baseHeight =
            height / worldSize;

        int remainder =
            height % worldSize;

        int localHeight =
            baseHeight;

        if (rank < remainder)
        {
            localHeight++;
        }

        vector<int> sendCounts;
        vector<int> displacements;

        if (rank == 0)
        {
            sendCounts.resize(
                worldSize
            );

            displacements.resize(
                worldSize
            );

            int offset = 0;

            for (int r = 0;
                 r < worldSize;
                 ++r)
            {
                int rankHeight =
                    baseHeight;

                if (r < remainder)
                {
                    rankHeight++;
                }

                sendCounts[r] =
                    rankHeight * width;

                displacements[r] =
                    offset;

                offset +=
                    rankHeight * width;
            }
        }

        vector<unsigned char>
            localImage(
                localHeight * width
            );

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double scatterStart =
            MPI_Wtime();

        MPI_Scatterv(
            rank == 0
                ? image.data()
                : nullptr,

            rank == 0
                ? sendCounts.data()
                : nullptr,

            rank == 0
                ? displacements.data()
                : nullptr,

            MPI_UNSIGNED_CHAR,

            localImage.data(),

            localHeight * width,

            MPI_UNSIGNED_CHAR,

            0,

            MPI_COMM_WORLD
        );

        double scatterEnd =
            MPI_Wtime();

        int extendedHeight =
            localHeight +
            2 * radius;

        vector<unsigned char>
            extended(
                extendedHeight * width,
                0
            );

        memcpy(
            extended.data() +
                radius * width,

            localImage.data(),

            localHeight * width
        );

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double haloStart =
            MPI_Wtime();

        exchangeStripHalos(
            extended,
            width,
            localHeight,
            radius,
            rank,
            worldSize
        );

        double haloEnd =
            MPI_Wtime();

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double computeStart =
            MPI_Wtime();

        vector<unsigned char>
            localOutput;

        boxFilterLocalStrip(
            extended,
            localOutput,
            width,
            localHeight,
            radius
        );

        double computeEnd =
            MPI_Wtime();

        vector<unsigned char>
            output;

        if (rank == 0)
        {
            output.resize(
                width * height
            );
        }

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double gatherStart =
            MPI_Wtime();

        MPI_Gatherv(
            localOutput.data(),

            localHeight * width,

            MPI_UNSIGNED_CHAR,

            rank == 0
                ? output.data()
                : nullptr,

            rank == 0
                ? sendCounts.data()
                : nullptr,

            rank == 0
                ? displacements.data()
                : nullptr,

            MPI_UNSIGNED_CHAR,

            0,

            MPI_COMM_WORLD
        );

        double gatherEnd =
            MPI_Wtime();

        double outputIOStart =
            MPI_Wtime();

        if (rank == 0)
        {
            createOutputDirectory();

            Mat outputImage(
                height,
                width,
                CV_8UC1,
                output.data()
            );

            imwrite(
                "output/mpi_output.jpg",
                outputImage
            );
        }

        double outputIOEnd =
            MPI_Wtime();

        double scatterTime =
            scatterEnd - scatterStart;

        double haloTime =
            haloEnd - haloStart;

        double computeTime =
            computeEnd - computeStart;

        double gatherTime =
            gatherEnd - gatherStart;

        double outputIOTime =
            outputIOEnd - outputIOStart;

        double maxScatter;
        double maxHalo;
        double maxCompute;
        double maxGather;
        double maxOutputIO;

        MPI_Reduce(
            &scatterTime,
            &maxScatter,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &haloTime,
            &maxHalo,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &computeTime,
            &maxCompute,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &gatherTime,
            &maxGather,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &outputIOTime,
            &maxOutputIO,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0)
        {
            cout
                << "Scatter time: "
                << maxScatter * 1000.0
                << " ms"
                << endl;

            cout
                << "Halo exchange: "
                << maxHalo * 1000.0
                << " ms"
                << endl;

            cout
                << "Filter computation: "
                << maxCompute * 1000.0
                << " ms"
                << endl;

            cout
                << "Gather time: "
                << maxGather * 1000.0
                << " ms"
                << endl;

            cout
                << "Output I/O: "
                << maxOutputIO * 1000.0
                << " ms"
                << endl;

            cout
                << "Output: "
                << "output/mpi_output.jpg"
                << endl;

            cout
                << "Input I/O: "
                << (inputEnd - inputStart)
                    * 1000.0
                << " ms"
                << endl;
        }
    }


    // ========================================================
    // 2D BLOCK MODE
    // ========================================================

    else if (mode == "block")
    {
        if (rank == 0)
        {
            cout
                << "-------------------------------------"
                << endl;

            cout
                << "2D BLOCK DECOMPOSITION"
                << endl;

            cout
                << "-------------------------------------"
                << endl;
        }

        // ----------------------------------------------------
        // CARTESIAN GRID
        // ----------------------------------------------------

        int dims[2] =
        {
            0,
            0
        };

        MPI_Dims_create(
            worldSize,
            2,
            dims
        );

        int periods[2] =
        {
            0,
            0
        };

        MPI_Comm cartComm;

        MPI_Cart_create(
            MPI_COMM_WORLD,
            2,
            dims,
            periods,
            0,
            &cartComm
        );

        int cartRank;

        MPI_Comm_rank(
            cartComm,
            &cartRank
        );

        int coords[2];

        MPI_Cart_coords(
            cartComm,
            cartRank,
            2,
            coords
        );

        // ----------------------------------------------------
        // NEIGHBORS
        // ----------------------------------------------------

        int topRank;
        int bottomRank;
        int leftRank;
        int rightRank;

        MPI_Cart_shift(
            cartComm,
            0,
            1,
            &topRank,
            &bottomRank
        );

        MPI_Cart_shift(
            cartComm,
            1,
            1,
            &leftRank,
            &rightRank
        );

        // ----------------------------------------------------
        // LOCAL BLOCK
        // ----------------------------------------------------

        int localWidth;
        int localHeight;

        int startX;
        int startY;

        calculateBlockSize(
            width,
            height,
            dims,
            coords,
            localWidth,
            localHeight,
            startX,
            startY
        );

        if (rank == 0)
        {
            cout
                << "Cartesian grid: "
                << dims[0]
                << " x "
                << dims[1]
                << endl;
        }

        // ----------------------------------------------------
        // PACK BLOCKS
        // ----------------------------------------------------

        vector<unsigned char>
            packedInput;

        vector<int>
            sendCounts;

        vector<int>
            displacements;

        if (rank == 0)
        {
            pack2DBlocks(
                image,
                packedInput,
                width,
                height,
                dims
            );

            sendCounts.resize(
                worldSize
            );

            displacements.resize(
                worldSize
            );

            int offset = 0;

            for (int r = 0;
                 r < worldSize;
                 ++r)
            {
                int rcoords[2];

                rcoords[0] =
                    r / dims[1];

                rcoords[1] =
                    r % dims[1];

                int rWidth;
                int rHeight;
                int rStartX;
                int rStartY;

                calculateBlockSize(
                    width,
                    height,
                    dims,
                    rcoords,
                    rWidth,
                    rHeight,
                    rStartX,
                    rStartY
                );

                sendCounts[r] =
                    rWidth *
                    rHeight;

                displacements[r] =
                    offset;

                offset +=
                    sendCounts[r];
            }
        }

        // ----------------------------------------------------
        // SCATTERV
        // ----------------------------------------------------

        vector<unsigned char>
            localImage(
                localWidth *
                localHeight
            );

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double scatterStart =
            MPI_Wtime();

        MPI_Scatterv(
            rank == 0
                ? packedInput.data()
                : nullptr,

            rank == 0
                ? sendCounts.data()
                : nullptr,

            rank == 0
                ? displacements.data()
                : nullptr,

            MPI_UNSIGNED_CHAR,

            localImage.data(),

            localWidth *
                localHeight,

            MPI_UNSIGNED_CHAR,

            0,

            MPI_COMM_WORLD
        );

        double scatterEnd =
            MPI_Wtime();

        // ----------------------------------------------------
        // EXTENDED BLOCK
        // ----------------------------------------------------

        int extendedWidth =
            localWidth +
            2 * radius;

        int extendedHeight =
            localHeight +
            2 * radius;

        vector<unsigned char>
            extended(
                extendedWidth *
                extendedHeight,
                0
            );

        for (int y = 0;
             y < localHeight;
             ++y)
        {
            memcpy(
                extended.data() +
                    (y + radius) *
                        extendedWidth +
                    radius,

                localImage.data() +
                    y * localWidth,

                localWidth
            );
        }

        // ----------------------------------------------------
        // HALO EXCHANGE
        // ----------------------------------------------------

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double haloStart =
            MPI_Wtime();

        exchangeBlockHalos(
            extended,
            localWidth,
            localHeight,
            radius,
            cartComm,
            leftRank,
            rightRank,
            topRank,
            bottomRank
        );

        double haloEnd =
            MPI_Wtime();

        // ----------------------------------------------------
        // COMPUTE
        // ----------------------------------------------------

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double computeStart =
            MPI_Wtime();

        vector<unsigned char>
            localOutput(
                localWidth *
                localHeight
            );

        boxFilterLocalBlock(
            extended,
            localOutput,
            localWidth,
            localHeight,
            radius
        );

        double computeEnd =
            MPI_Wtime();

        // ----------------------------------------------------
        // GATHERV
        // ----------------------------------------------------

        vector<unsigned char>
            packedOutput;

        // FIX:
        // The final global output must also be declared
        // inside block mode.
        vector<unsigned char>
            output;

        if (rank == 0)
        {
            packedOutput.resize(
                width * height
            );
        }

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double gatherStart =
            MPI_Wtime();

        MPI_Gatherv(
            localOutput.data(),

            localWidth *
                localHeight,

            MPI_UNSIGNED_CHAR,

            rank == 0
                ? packedOutput.data()
                : nullptr,

            rank == 0
                ? sendCounts.data()
                : nullptr,

            rank == 0
                ? displacements.data()
                : nullptr,

            MPI_UNSIGNED_CHAR,

            0,

            MPI_COMM_WORLD
        );

        double gatherEnd =
            MPI_Wtime();

        // ----------------------------------------------------
        // UNPACK + OUTPUT
        // ----------------------------------------------------

        double outputIOStart =
            MPI_Wtime();

        if (rank == 0)
        {
            output.resize(
                width * height
            );

            unpack2DBlocks(
                packedOutput,
                output,
                width,
                height,
                dims
            );

            createOutputDirectory();

            Mat outputImage(
                height,
                width,
                CV_8UC1,
                output.data()
            );

            if (!imwrite(
                    "output/mpi_output.jpg",
                    outputImage))
            {
                cerr
                    << "ERROR: Failed to write "
                       "output image."
                    << endl;
            }
        }

        double outputIOEnd =
            MPI_Wtime();

        // ----------------------------------------------------
        // TIMINGS
        // ----------------------------------------------------

        double scatterTime =
            scatterEnd -
            scatterStart;

        double haloTime =
            haloEnd -
            haloStart;

        double computeTime =
            computeEnd -
            computeStart;

        double gatherTime =
            gatherEnd -
            gatherStart;

        double outputIOTime =
            outputIOEnd -
            outputIOStart;

        double maxScatter = 0.0;
        double maxHalo = 0.0;
        double maxCompute = 0.0;
        double maxGather = 0.0;
        double maxOutputIO = 0.0;

        MPI_Reduce(
            &scatterTime,
            &maxScatter,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &haloTime,
            &maxHalo,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &computeTime,
            &maxCompute,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &gatherTime,
            &maxGather,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &outputIOTime,
            &maxOutputIO,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0)
        {
            cout
                << "Cartesian grid: "
                << dims[0]
                << " x "
                << dims[1]
                << endl;

            cout
                << "Local block: "
                << localWidth
                << " x "
                << localHeight
                << endl;

            cout
                << "Scatter time: "
                << maxScatter * 1000.0
                << " ms"
                << endl;

            cout
                << "Halo exchange: "
                << maxHalo * 1000.0
                << " ms"
                << endl;

            cout
                << "Filter computation: "
                << maxCompute * 1000.0
                << " ms"
                << endl;

            cout
                << "Gather time: "
                << maxGather * 1000.0
                << " ms"
                << endl;

            cout
                << "Output I/O: "
                << maxOutputIO * 1000.0
                << " ms"
                << endl;

            cout
                << "Output: "
                << "output/mpi_output.jpg"
                << endl;

            cout
                << "Input I/O: "
                << (inputEnd - inputStart)
                    * 1000.0
                << " ms"
                << endl;
        }

        MPI_Comm_free(
            &cartComm
        );
    }
    // ========================================================
    // OVERLAP MODE
    // ========================================================

    else if (mode == "overlap")
    {
        if (rank == 0)
        {
            cout
                << "-------------------------------------"
                << endl;

            cout
                << "2D BLOCK WITH OVERLAPPED HALO EXCHANGE"
                << endl;

            cout
                << "-------------------------------------"
                << endl;
        }

        // ----------------------------------------------------
        // CARTESIAN GRID
        // ----------------------------------------------------

        int dims[2] =
        {
            0,
            0
        };

        MPI_Dims_create(
            worldSize,
            2,
            dims
        );

        int periods[2] =
        {
            0,
            0
        };

        MPI_Comm cartComm;

        MPI_Cart_create(
            MPI_COMM_WORLD,
            2,
            dims,
            periods,
            0,
            &cartComm
        );

        int cartRank;

        MPI_Comm_rank(
            cartComm,
            &cartRank
        );

        int coords[2];

        MPI_Cart_coords(
            cartComm,
            cartRank,
            2,
            coords
        );

        // ----------------------------------------------------
        // NEIGHBORS
        // ----------------------------------------------------

        int topRank;
        int bottomRank;
        int leftRank;
        int rightRank;

        MPI_Cart_shift(
            cartComm,
            0,
            1,
            &topRank,
            &bottomRank
        );

        MPI_Cart_shift(
            cartComm,
            1,
            1,
            &leftRank,
            &rightRank
        );

        // ----------------------------------------------------
        // LOCAL BLOCK
        // ----------------------------------------------------

        int localWidth;
        int localHeight;

        int startX;
        int startY;

        calculateBlockSize(
            width,
            height,
            dims,
            coords,
            localWidth,
            localHeight,
            startX,
            startY
        );

        // ----------------------------------------------------
        // PACK BLOCKS
        // ----------------------------------------------------

        vector<unsigned char>
            packedInput;

        vector<int>
            sendCounts;

        vector<int>
            displacements;

        if (rank == 0)
        {
            pack2DBlocks(
                image,
                packedInput,
                width,
                height,
                dims
            );

            sendCounts.resize(
                worldSize
            );

            displacements.resize(
                worldSize
            );

            int offset = 0;

            for (int r = 0;
                 r < worldSize;
                 ++r)
            {
                int rcoords[2];

                rcoords[0] =
                    r / dims[1];

                rcoords[1] =
                    r % dims[1];

                int rWidth;
                int rHeight;
                int rStartX;
                int rStartY;

                calculateBlockSize(
                    width,
                    height,
                    dims,
                    rcoords,
                    rWidth,
                    rHeight,
                    rStartX,
                    rStartY
                );

                sendCounts[r] =
                    rWidth *
                    rHeight;

                displacements[r] =
                    offset;

                offset +=
                    sendCounts[r];
            }
        }

        // ----------------------------------------------------
        // SCATTERV
        // ----------------------------------------------------

        vector<unsigned char>
            localImage(
                localWidth *
                localHeight
            );

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double scatterStart =
            MPI_Wtime();

        MPI_Scatterv(
            rank == 0
                ? packedInput.data()
                : nullptr,

            rank == 0
                ? sendCounts.data()
                : nullptr,

            rank == 0
                ? displacements.data()
                : nullptr,

            MPI_UNSIGNED_CHAR,

            localImage.data(),

            localWidth *
                localHeight,

            MPI_UNSIGNED_CHAR,

            0,

            MPI_COMM_WORLD
        );

        double scatterEnd =
            MPI_Wtime();

        // ----------------------------------------------------
        // EXTENDED BLOCK
        // ----------------------------------------------------

        int extendedWidth =
            localWidth +
            2 * radius;

        int extendedHeight =
            localHeight +
            2 * radius;

        vector<unsigned char>
            extended(
                extendedWidth *
                extendedHeight,
                0
            );

        for (int y = 0;
             y < localHeight;
             ++y)
        {
            memcpy(
                extended.data() +
                    (y + radius) *
                        extendedWidth +
                    radius,

                localImage.data() +
                    y * localWidth,

                localWidth
            );
        }

        vector<unsigned char>
            localOutput(
                localWidth *
                localHeight,
                0
            );

        // ----------------------------------------------------
        // OVERLAPPED HALO + COMPUTATION
        //
        // Phase 1:
        //   Start left/right communication.
        //   Compute the strict interior while it is in flight.
        //
        // Phase 2:
        //   Wait for left/right halos.
        //   Start top/bottom communication.
        //   Compute rows that do not depend on top/bottom halos.
        //
        // Phase 3:
        //   Wait for top/bottom halos.
        //   Compute the remaining border rows.
        // ----------------------------------------------------

        MPI_Datatype columnType;

        MPI_Type_vector(
            extendedHeight,
            radius,
            extendedWidth,
            MPI_UNSIGNED_CHAR,
            &columnType
        );

        MPI_Type_commit(
            &columnType
        );

        MPI_Request horizontalRequests[4];

        double processingStart =
            MPI_Wtime();

        // ----------------------------------------------------
        // PHASE 1: LEFT / RIGHT NON-BLOCKING EXCHANGE
        // ----------------------------------------------------

        MPI_Irecv(
            extended.data(),
            1,
            columnType,
            leftRank,
            300,
            cartComm,
            &horizontalRequests[0]
        );

        MPI_Irecv(
            extended.data() +
                radius +
                localWidth,
            1,
            columnType,
            rightRank,
            301,
            cartComm,
            &horizontalRequests[1]
        );

        MPI_Isend(
            extended.data() +
                radius,
            1,
            columnType,
            leftRank,
            301,
            cartComm,
            &horizontalRequests[2]
        );

        MPI_Isend(
            extended.data() +
                localWidth,
            1,
            columnType,
            rightRank,
            300,
            cartComm,
            &horizontalRequests[3]
        );

        // ----------------------------------------------------
        // COMPUTE STRICT INTERIOR WHILE HORIZONTAL HALOS
        // ARE IN FLIGHT.
        // ----------------------------------------------------

        double computeWorkStart =
            MPI_Wtime();

        int interiorYStart =
            radius;

        int interiorYEnd =
            localHeight - radius;

        int interiorXStart =
            radius;

        int interiorXEnd =
            localWidth - radius;

        if (interiorYStart < interiorYEnd &&
            interiorXStart < interiorXEnd)
        {
            boxFilterLocalBlockRegion(
                extended,
                localOutput,
                localWidth,
                localHeight,
                radius,
                interiorYStart,
                interiorYEnd,
                interiorXStart,
                interiorXEnd
            );
        }

        double computeWorkEnd =
            MPI_Wtime();

        double computeWorkTime =
            computeWorkEnd -
            computeWorkStart;

        // ----------------------------------------------------
        // WAIT FOR HORIZONTAL HALOS
        // ----------------------------------------------------

        double horizontalWaitStart =
            MPI_Wtime();

        MPI_Waitall(
            4,
            horizontalRequests,
            MPI_STATUSES_IGNORE
        );

        double horizontalWaitEnd =
            MPI_Wtime();

        double communicationWaitTime =
            horizontalWaitEnd -
            horizontalWaitStart;

        // ----------------------------------------------------
        // GLOBAL LEFT / RIGHT BOUNDARIES
        // ----------------------------------------------------

        if (leftRank == MPI_PROC_NULL)
        {
            for (int y = 0;
                 y < extendedHeight;
                 ++y)
            {
                unsigned char value =
                    extended[
                        y * extendedWidth +
                        radius
                    ];

                for (int x = 0;
                     x < radius;
                     ++x)
                {
                    extended[
                        y * extendedWidth +
                        x
                    ] = value;
                }
            }
        }

        if (rightRank == MPI_PROC_NULL)
        {
            for (int y = 0;
                 y < extendedHeight;
                 ++y)
            {
                unsigned char value =
                    extended[
                        y * extendedWidth +
                        radius +
                        localWidth - 1
                    ];

                for (int x = 0;
                     x < radius;
                     ++x)
                {
                    extended[
                        y * extendedWidth +
                        radius +
                        localWidth +
                        x
                    ] = value;
                }
            }
        }

        // ----------------------------------------------------
        // PHASE 2: TOP / BOTTOM NON-BLOCKING EXCHANGE
        //
        // Horizontal halos are now available, so the rows
        // sent here include correct corner data.
        // ----------------------------------------------------

        MPI_Request verticalRequests[4];

        int rowWidth =
            extendedWidth;

        unsigned char* topHalo =
            extended.data();

        unsigned char* firstActualRows =
            extended.data() +
            radius * extendedWidth;

        unsigned char* bottomHalo =
            extended.data() +
            (radius + localHeight) *
                extendedWidth;

        unsigned char* lastActualRows =
            extended.data() +
            localHeight *
                extendedWidth;

        MPI_Irecv(
            topHalo,
            radius * rowWidth,
            MPI_UNSIGNED_CHAR,
            topRank,
            400,
            cartComm,
            &verticalRequests[0]
        );

        MPI_Irecv(
            bottomHalo,
            radius * rowWidth,
            MPI_UNSIGNED_CHAR,
            bottomRank,
            401,
            cartComm,
            &verticalRequests[1]
        );

        MPI_Isend(
            firstActualRows,
            radius * rowWidth,
            MPI_UNSIGNED_CHAR,
            topRank,
            401,
            cartComm,
            &verticalRequests[2]
        );

        MPI_Isend(
            lastActualRows,
            radius * rowWidth,
            MPI_UNSIGNED_CHAR,
            bottomRank,
            400,
            cartComm,
            &verticalRequests[3]
        );

        // ----------------------------------------------------
        // COMPUTE MIDDLE ROWS WHILE TOP/BOTTOM HALOS
        // ARE IN FLIGHT.
        // ----------------------------------------------------

        double computeMiddleStart =
            MPI_Wtime();

        int middleYStart =
            radius;

        int middleYEnd =
            localHeight - radius;

        if (middleYStart < middleYEnd)
        {
            boxFilterLocalBlockRegion(
                extended,
                localOutput,
                localWidth,
                localHeight,
                radius,
                middleYStart,
                middleYEnd,
                0,
                localWidth
            );
        }

        double computeMiddleEnd =
            MPI_Wtime();

        computeWorkTime +=
            computeMiddleEnd -
            computeMiddleStart;

        // ----------------------------------------------------
        // WAIT FOR TOP / BOTTOM HALOS
        // ----------------------------------------------------

        double verticalWaitStart =
            MPI_Wtime();

        MPI_Waitall(
            4,
            verticalRequests,
            MPI_STATUSES_IGNORE
        );

        double verticalWaitEnd =
            MPI_Wtime();

        communicationWaitTime +=
            verticalWaitEnd -
            verticalWaitStart;

        // ----------------------------------------------------
        // GLOBAL TOP / BOTTOM BOUNDARIES
        // ----------------------------------------------------

        if (topRank == MPI_PROC_NULL)
        {
            unsigned char* firstActual =
                extended.data() +
                radius * extendedWidth;

            for (int r = 0;
                 r < radius;
                 ++r)
            {
                memcpy(
                    extended.data() +
                        r * extendedWidth,

                    firstActual,

                    extendedWidth
                );
            }
        }

        if (bottomRank == MPI_PROC_NULL)
        {
            unsigned char* lastActual =
                extended.data() +
                (radius +
                 localHeight - 1) *
                    extendedWidth;

            for (int r = 0;
                 r < radius;
                 ++r)
            {
                memcpy(
                    extended.data() +
                        (radius +
                         localHeight +
                         r) *
                            extendedWidth,

                    lastActual,

                    extendedWidth
                );
            }
        }

        // ----------------------------------------------------
        // PHASE 3: COMPUTE TOP / BOTTOM BORDER ROWS
        // ----------------------------------------------------

        double computeBorderStart =
            MPI_Wtime();

        int topBorderEnd =
            min(radius, localHeight);

        if (topBorderEnd > 0)
        {
            boxFilterLocalBlockRegion(
                extended,
                localOutput,
                localWidth,
                localHeight,
                radius,
                0,
                topBorderEnd,
                0,
                localWidth
            );
        }

        int bottomBorderStart =
            max(
                topBorderEnd,
                localHeight - radius
            );

        if (bottomBorderStart < localHeight)
        {
            boxFilterLocalBlockRegion(
                extended,
                localOutput,
                localWidth,
                localHeight,
                radius,
                bottomBorderStart,
                localHeight,
                0,
                localWidth
            );
        }

        double computeBorderEnd =
            MPI_Wtime();

        computeWorkTime +=
            computeBorderEnd -
            computeBorderStart;

        double processingEnd =
            MPI_Wtime();

        MPI_Type_free(
            &columnType
        );

        double overlapProcessingTime =
            processingEnd -
            processingStart;

        // ----------------------------------------------------
        // GATHERV
        // ----------------------------------------------------

        vector<unsigned char>
            packedOutput;

        vector<unsigned char>
            output;

        if (rank == 0)
        {
            packedOutput.resize(
                width * height
            );
        }

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        double gatherStart =
            MPI_Wtime();

        MPI_Gatherv(
            localOutput.data(),

            localWidth *
                localHeight,

            MPI_UNSIGNED_CHAR,

            rank == 0
                ? packedOutput.data()
                : nullptr,

            rank == 0
                ? sendCounts.data()
                : nullptr,

            rank == 0
                ? displacements.data()
                : nullptr,

            MPI_UNSIGNED_CHAR,

            0,

            MPI_COMM_WORLD
        );

        double gatherEnd =
            MPI_Wtime();

        // ----------------------------------------------------
        // UNPACK + OUTPUT
        // ----------------------------------------------------

        double outputIOStart =
            MPI_Wtime();

        if (rank == 0)
        {
            output.resize(
                width * height
            );

            unpack2DBlocks(
                packedOutput,
                output,
                width,
                height,
                dims
            );

            createOutputDirectory();

            Mat outputImage(
                height,
                width,
                CV_8UC1,
                output.data()
            );

            if (!imwrite(
                    "output/mpi_output.jpg",
                    outputImage))
            {
                cerr
                    << "ERROR: Failed to write "
                       "output image."
                    << endl;
            }
        }

        double outputIOEnd =
            MPI_Wtime();

        // ----------------------------------------------------
        // TIMINGS
        // ----------------------------------------------------

        double scatterTime =
            scatterEnd -
            scatterStart;

        double gatherTime =
            gatherEnd -
            gatherStart;

        double outputIOTime =
            outputIOEnd -
            outputIOStart;

        double maxScatter = 0.0;
        double maxOverlap = 0.0;
        double maxCompute = 0.0;
        double maxCommWait = 0.0;
        double maxGather = 0.0;
        double maxOutputIO = 0.0;

        MPI_Reduce(
            &scatterTime,
            &maxScatter,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &overlapProcessingTime,
            &maxOverlap,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &computeWorkTime,
            &maxCompute,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &communicationWaitTime,
            &maxCommWait,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &gatherTime,
            &maxGather,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &outputIOTime,
            &maxOutputIO,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0)
        {
            cout
                << "Cartesian grid: "
                << dims[0]
                << " x "
                << dims[1]
                << endl;

            cout
                << "Local block: "
                << localWidth
                << " x "
                << localHeight
                << endl;

            cout
                << "Scatter time: "
                << maxScatter * 1000.0
                << " ms"
                << endl;

            cout
                << "Overlap processing: "
                << maxOverlap * 1000.0
                << " ms"
                << endl;

            cout
                << "Filter computation: "
                << maxCompute * 1000.0
                << " ms"
                << endl;

            cout
                << "Communication wait: "
                << maxCommWait * 1000.0
                << " ms"
                << endl;

            cout
                << "Gather time: "
                << maxGather * 1000.0
                << " ms"
                << endl;

            cout
                << "Output I/O: "
                << maxOutputIO * 1000.0
                << " ms"
                << endl;

            cout
                << "Output: "
                << "output/mpi_output.jpg"
                << endl;

            cout
                << "Input I/O: "
                << (inputEnd - inputStart)
                    * 1000.0
                << " ms"
                << endl;
        }

        MPI_Comm_free(
            &cartComm
        );
    }
    else
    {
        if (rank == 0)
        {
            cout
                << "ERROR: Unknown mode '"
                << mode
                << "'."
                << endl;

            cout
                << "Use --mode strip, --mode block, "
                   "or --mode overlap."
                << endl;
        }
    }

    MPI_Finalize();

    return 0;
}