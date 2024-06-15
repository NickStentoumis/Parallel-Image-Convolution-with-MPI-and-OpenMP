# Parallel-Image-Convolution-with-MPI-and-OpenMP

This program performs image convolution using MPI for parallel processing and OpenMP for multi-threading within each process. It supports both greyscale and RGB image types.

## Usage

### Execution: 
* Run the program using MPI
mpiexec -n <num_processes> ./main <image_file> <width> <height> <type> <steps>
* <image_file>: Path to the input image file.
* <width>: Width of the image in pixels.
* <height>: Height of the image in pixels.
* <type>: Image type (grey for greyscale, rgb for RGB).
* <steps>: Number of convolution steps to perform.

### Output:
The processed image will be saved as convolutionImage.raw in the current directory.

## Implementation Details
MPI Communication: Processes exchange image data and synchronize computations using MPI.

OpenMP Parallelism: Within each MPI process, OpenMP parallelizes the convolution operation across multiple threads for improved performance.

File I/O: MPI is used for efficient reading and writing of image data (MPI_File_read and MPI_File_write).

Performance: Timing measurements (MPI_Wtime) are used to evaluate the total execution time, with process 0 determining and printing the maximum time among all processes.

### Dependencies
MPI and OpenMP: Ensure MPI and OpenMP are installed and configured correctly on your system.

### Notes
Ensure the input image file exists and is accessible to all MPI processes.
Adjust compilation settings (Makefile) based on your MPI setup (mpicc, mpiexec, etc.).