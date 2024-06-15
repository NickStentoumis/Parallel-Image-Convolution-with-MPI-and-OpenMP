#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"
#include <stdint.h>
#include <fcntl.h>
#include "omp.h"

int divideRows(int, int, int);
uint8_t *returnPointer(uint8_t *, int, int, int);
void convoluteGrey(uint8_t *, uint8_t *, int, int, int, int, float h[3][3]);
void convoluteRgb(uint8_t *, uint8_t *, int, int, int, int, float h[3][3]);

int main(int argc, char** argv){
	int width, height, steps, rowDivision, colDivision, rows, columns, type, processId, numOfProcesses;
	double timer, otherProcessTime;
	char* imageName = malloc(100*sizeof(char));

	MPI_Init(&argc, &argv);//mpi initialization
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);
	MPI_Comm_rank(MPI_COMM_WORLD, &processId);
	MPI_Status status;

	

	MPI_Datatype greyCol, rgbCol, greyRow, rgbRow;
	MPI_Request SendNorth, SendSouth, SendEast, SendWest, RecvNorth, RecvSouth, RecvWest, RecvEast;

	int north = -5, south = -5, east = -5, west = -5;
	
	strcpy(imageName, argv[1]);
	if(processId == 0){
		if(argc == 6 && strcmp(argv[4], "grey") == 0){
			width = atoi(argv[2]);
			height = atoi(argv[3]);
			steps = atoi(argv[5]);
			type = 1;
		}
		else if(argc == 6 && strcmp(argv[4], "rgb") == 0){
			width = atoi(argv[2]);
			height = atoi(argv[3]);
			steps = atoi(argv[5]);
			type = 2;
		}
		else{
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			exit(EXIT_FAILURE);
		}
		rowDivision = divideRows(height, width, numOfProcesses);
		colDivision = numOfProcesses / rowDivision;
	}

	MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);//broadcast to all processes
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rowDivision, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colDivision, 1, MPI_INT, 0, MPI_COMM_WORLD);

    rows = height / rowDivision;//calculate rows and columns the process will have
    columns = width / colDivision;

    MPI_Type_vector(rows, 1, columns + 2, MPI_BYTE, &greyCol);//datatypes for send and recv
	MPI_Type_contiguous(columns, MPI_BYTE, &greyRow);
	MPI_Type_commit(&greyCol);
	MPI_Type_commit(&greyRow);
	MPI_Type_vector(rows, 3, 3 * (columns + 2), MPI_BYTE, &rgbCol);
	MPI_Type_contiguous(3 * columns, MPI_BYTE, &rgbRow);
	MPI_Type_commit(&rgbCol);
	MPI_Type_commit(&rgbRow);

	int firstRow = (processId / colDivision) * rows;//calculate where the process will start reading in parallel i/o
	int firstColumn = (processId % colDivision) * columns;

	float h[3][3] = {{(float) 1/16, (float) 2/16, (float) 1/16}, {(float) 2/16, (float) 4/16, (float) 2/16}, { (float) 1/16, (float) 2/16, (float) 1/16}};

	uint8_t *source = NULL, *destination = NULL, *tempArray = NULL, *arrayPointer = NULL;
	
	if (type == 1){
		source = malloc((rows + 2) * (columns + 2) * sizeof(uint8_t));
		destination = malloc((rows + 2) * (columns + 2) * sizeof(uint8_t));
	}
	else if (type == 2){
		source = malloc((rows + 2) * (columns*3 + 6) * sizeof(uint8_t));
		destination = malloc((rows + 2) * (columns*3 + 6) * sizeof(uint8_t));
	}

	MPI_File mpifile;
	int counter = 0;

	MPI_File_open(MPI_COMM_WORLD, imageName, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpifile);//parallel reading in file
	if (type == 1) {
		for (counter = 1 ; counter <= rows ; counter++) {
			MPI_File_seek(mpifile, (firstRow + counter-1) * width + firstColumn, MPI_SEEK_SET);
			arrayPointer = returnPointer(source, counter, 1, columns+2);
			MPI_File_read(mpifile, arrayPointer, columns, MPI_BYTE, &status);
		}
	} 
	else if (type == 2) {
		for (counter = 1 ; counter <= rows ; counter++) {
			MPI_File_seek(mpifile, 3*(firstRow + counter-1) * width + 3*firstColumn, MPI_SEEK_SET);
			arrayPointer = returnPointer(source, counter, 3, columns*3+6);
			MPI_File_read(mpifile, arrayPointer, columns*3, MPI_BYTE, &status);
		}
	}

	MPI_File_close(&mpifile);

	if (firstRow != 0){//calculate neighbours
		north = processId - colDivision;
	}
	if (firstRow + rows != height){
		south = processId + colDivision;
	}
	if (firstColumn != 0){
		west = processId - 1;
	}
	if(firstColumn + columns != width){
		east = processId + 1;
	}

	
	MPI_Barrier(MPI_COMM_WORLD);//every process needs to be here before main loop

    timer = MPI_Wtime();

    int loopCounter = 0;
    for (loopCounter = 0; loopCounter < steps; loopCounter++){
    	if (type == 1){
    		if (north != -5){//send and receive data 
    			MPI_Isend(returnPointer(source, 1, 1, columns+2), 1, greyRow, north, 0, MPI_COMM_WORLD, &SendNorth);
    			MPI_Irecv(returnPointer(source, 0, 1, columns+2), 1, greyRow, north, 0, MPI_COMM_WORLD, &RecvNorth);
    		}
    		if (west != -5){
    			MPI_Isend(returnPointer(source, 1, 1, columns+2), 1, greyCol, west, 0, MPI_COMM_WORLD, &SendWest);
    			MPI_Irecv(returnPointer(source, 1, 0, columns+2), 1, greyCol, west, 0, MPI_COMM_WORLD, &RecvWest);
    		}
    		if (south != -5){
    			MPI_Isend(returnPointer(source, rows, 1, columns+2), 1, greyRow, south, 0, MPI_COMM_WORLD, &SendSouth);
    			MPI_Irecv(returnPointer(source, rows+1, 1, columns+2), 1, greyRow, south, 0, MPI_COMM_WORLD, &RecvSouth);
    		}
    		if (east != -5){
    			MPI_Isend(returnPointer(source, 1, columns, columns+2), 1, greyCol, east, 0, MPI_COMM_WORLD, &SendEast);
    			MPI_Irecv(returnPointer(source, 1, columns+1, columns+2), 1, greyCol, east, 0, MPI_COMM_WORLD, &RecvEast);
    		}
    	}
    	else if (type == 2){
    		if (north != -5){
    			MPI_Isend(returnPointer(source, 1, 3, 3*(columns+2)), 1, rgbRow, north, 0, MPI_COMM_WORLD, &SendNorth);
    			MPI_Irecv(returnPointer(source, 0, 3, 3*(columns+2)), 1, rgbRow, north, 0, MPI_COMM_WORLD, &RecvNorth);
    		}
    		if (west != -5){
    			MPI_Isend(returnPointer(source, 1, 3, 3*(columns+2)), 1, rgbCol, west, 0, MPI_COMM_WORLD, &SendWest);
    			MPI_Irecv(returnPointer(source, 1, 0, 3*(columns+2)), 1, rgbCol, west, 0, MPI_COMM_WORLD, &RecvWest);
    		}
    		if (south != -5){
    			MPI_Isend(returnPointer(source, rows, 3, 3*(columns+2)), 1, rgbRow, south, 0, MPI_COMM_WORLD, &SendSouth);
    			MPI_Irecv(returnPointer(source, rows+1, 3, 3*(columns+2)), 1, rgbRow, south, 0, MPI_COMM_WORLD, &RecvSouth);
    		}
    		if (east != -5){
    			MPI_Isend(returnPointer(source, 1, 3*columns, 3*(columns+2)), 1, rgbCol, east, 0, MPI_COMM_WORLD, &SendEast);
    			MPI_Irecv(returnPointer(source, 1, 3*(columns+1), 3*(columns+2)), 1, rgbCol, east, 0, MPI_COMM_WORLD, &RecvEast);
    		}
    	}
    	int cnt1, cnt2;
		if (type == 1) {//inner calculations
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)			
			for (cnt1 = 1 ; cnt1 <= rows ; cnt1++){
				for (cnt2 = 1 ; cnt2 <= columns ; cnt2++){
					convoluteGrey(source, destination, cnt1, cnt2, columns+2, rows, h);
				}
			}
		} 
		else if (type == 2) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)			
			for (cnt1 = 1 ; cnt1 <= rows ; cnt1++){
				for (cnt2 = 1 ; cnt2 <= columns ; cnt2++){
					convoluteRgb(source, destination, cnt1, cnt2*3, columns*3+6, rows, h);
				}
			}
		}
		if (north != -5) {//outer calculations
			MPI_Wait(&RecvNorth, &status);
			if (type == 1) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)				
				for (cnt1 = 1 ; cnt1 <= 1 ; cnt1++){
					for (cnt2 = 2 ; cnt2 <= (columns - 1) ; cnt2++){
						convoluteGrey(source, destination, cnt1, cnt2, columns+2, rows, h);
					}
				}
			} 
			else if (type == 2) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)				
				for (cnt1 = 1 ; cnt1 <= 1 ; cnt1++){
					for (cnt2 = 2 ; cnt2 <= (columns - 1) ; cnt2++){
						convoluteRgb(source, destination, cnt1, cnt2*3, columns*3+6, rows, h);
					}
				}
			}
		}
		if (west != -5) {
			MPI_Wait(&RecvWest, &status);
			if (type == 1) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = 2 ; cnt1 <= (rows - 1) ; cnt1++){
					for (cnt2 = 1 ; cnt2 <= 1 ; cnt2++){
						convoluteGrey(source, destination, cnt1, cnt2, columns+2, rows, h);
					}
				}
			} 
			else if (type == 2) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = 1 ; cnt1 <= (rows - 1) ; cnt1++){
					for (cnt2 = 1 ; cnt2 <= 1 ; cnt2++){
						convoluteRgb(source, destination, cnt1, cnt2*3, columns*3+6, rows, h);
					}
				}
			}
		}
		if (south != -5) {
			MPI_Wait(&RecvSouth, &status);
			if (type == 1) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = rows ; cnt1 <= rows ; cnt1++){
					for (cnt2 = 2 ; cnt2 <= (columns - 1) ; cnt2++){
						convoluteGrey(source, destination, cnt1, cnt2, columns+2, rows, h);
					}
				}
			} 
			else if (type == 2) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = rows ; cnt1 <= rows ; cnt1++){
					for (cnt2 = 2 ; cnt2 <= (columns - 1) ; cnt2++){
						convoluteRgb(source, destination, cnt1, cnt2*3, columns*3+6, rows, h);
					}
				}
			}
		}
		if (east != -5) {
			MPI_Wait(&RecvEast, &status);
			if (type == 1) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = 2 ; cnt1 <= (rows - 1) ; cnt1++){
					for (cnt2 = columns ; cnt2 <= columns; cnt2++){
						convoluteGrey(source, destination, cnt1, cnt2, columns+2, rows, h);
					}
				}
			} 
			else if (type == 2) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = 2 ; cnt1 <= (rows - 1) ; cnt1++){
					for (cnt2 = columns ; cnt2 <= columns; cnt2++){
						convoluteRgb(source, destination, cnt1, cnt2*3, columns*3+6, rows, h);
					}
				}
			}
		}
		
		if (north != -5 && west != -5){//corners
			if (type == 1) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = 1 ; cnt1 <= 1 ; cnt1++){
					for (cnt2 = 1 ; cnt2 <= 1 ; cnt2++){
						convoluteGrey(source, destination, cnt1, cnt2, columns+2, rows, h);
					}
				}
			} 
			else if (type == 2) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = 1 ; cnt1 <= 1 ; cnt1++){
					for (cnt2 = 1 ; cnt2 <= 1 ; cnt2++){
						convoluteRgb(source, destination, cnt1, cnt2*3, columns*3+6, rows, h);
					}
				}
			}
		}
		if (west != -5 && south != -5){
			if (type == 1) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = rows ; cnt1 <= rows ; cnt1++){
					for (cnt2 = 1; cnt2 <= 1; cnt2++){
						convoluteGrey(source, destination, cnt1, cnt2, columns+2, rows, h);
					}
				}
			} 
			else if (type == 2) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = rows ; cnt1 <= rows ; cnt1++){
					for (cnt2 = 1 ; cnt2 <= 1; cnt2++){
						convoluteRgb(source, destination, cnt1, cnt2*3, columns*3+6, rows, h);
					}
				}
			}
		}
		if (south != -5 && east != -5){
			if (type == 1) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = rows ; cnt1 <= rows ; cnt1++){
					for (cnt2 = columns ; cnt2 <= columns; cnt2++){
						convoluteGrey(source, destination, cnt1, cnt2, columns+2, rows, h);
					}
				}
			} 
			else if (type == 2) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = rows; cnt1 <= rows ; cnt1++){
					for (cnt2 = columns ; cnt2 <= columns ; cnt2++){
						convoluteRgb(source, destination, cnt1, cnt2*3, columns*3+6, rows, h);
					}
				}
			}
		}
		if (east != -5 && north != -5){
			if (type == 1) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = 1 ; cnt1 <= 1 ; cnt1++){
					for (cnt2 = columns; cnt2 <= columns; cnt2++){
						convoluteGrey(source, destination, cnt1, cnt2, columns+2, rows, h);
					}
				}
			} 
			else if (type == 2) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
				for (cnt1 = 1 ; cnt1 <= 1 ; cnt1++){
					for (cnt2 = columns; cnt2 <= columns ; cnt2++){
						convoluteRgb(source, destination, cnt1, cnt2*3, columns*3+6, rows, h);
					}
				}
			}
		}
		if (north != -5){
			MPI_Wait(&SendNorth, &status);
		}
		if (west != -5){
			MPI_Wait(&SendWest, &status);
		}
		if (south != -5){
			MPI_Wait(&SendSouth, &status);
		}
		if (east != -5){
			MPI_Wait(&SendEast, &status);
		}

		tempArray = source;
        source = destination;
        destination = tempArray;
	}

    timer = MPI_Wtime() - timer;

    char *newImage = malloc((30) * sizeof(char));//parallel writing
	strcpy(newImage, "convolutionImage.raw");
	MPI_File newMpiFile;
	MPI_File_open(MPI_COMM_WORLD, newImage, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &newMpiFile);
	if (type == 1) {
		for (counter = 1 ; counter <= rows ; counter++) {
			MPI_File_seek(newMpiFile, (firstRow + counter-1) * width + firstColumn, MPI_SEEK_SET);
			arrayPointer = returnPointer(source, counter, 1, columns+2);
			MPI_File_write(newMpiFile, arrayPointer, columns, MPI_BYTE, MPI_STATUS_IGNORE);
		}
	} 
	else if (type == 2) {
		for (counter = 1 ; counter <= rows ; counter++) {
			MPI_File_seek(newMpiFile, 3*(firstRow + counter-1) * width + 3*firstColumn, MPI_SEEK_SET);
			arrayPointer = returnPointer(source, counter, 3, columns*3+6);
			MPI_File_write(newMpiFile, arrayPointer, columns*3, MPI_BYTE, MPI_STATUS_IGNORE);
		}
	}
	MPI_File_close(&newMpiFile);

    if (processId != 0){//get times from other processes and print maximum time
        MPI_Send(&timer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else {
        for (counter = 1 ; counter != numOfProcesses ; ++counter) {
            MPI_Recv(&otherProcessTime, 1, MPI_DOUBLE, counter, 0, MPI_COMM_WORLD, &status);
            if (otherProcessTime > timer){
                timer = otherProcessTime;
        	}
        }
        printf("%f\n", timer);
    }


	MPI_Type_free(&rgbRow);
	MPI_Type_free(&rgbCol);
	MPI_Type_free(&greyRow);
	MPI_Type_free(&greyCol);
	MPI_Finalize();
	return 0;
}


int divideRows(int rows, int columns, int processes) {
    int flg, rows_to, columns_to, bestdivide = 0;
    int perMin = rows + columns + 1;
    for (rows_to = 1 ; rows_to <= processes ; ++rows_to) {
        if (processes % rows_to || rows % rows_to) continue;
        columns_to = processes / rows_to;
        if (columns % columns_to) continue;
        flg = rows / rows_to + columns / columns_to;
        if (flg < perMin) {
            perMin = flg;
            bestdivide = rows_to;
        }
    }
    return bestdivide;
}

uint8_t *returnPointer(uint8_t *array, int i, int j, int width) {
    return &array[width * i + j];
}

void convoluteGrey(uint8_t *source, uint8_t *destination, int c1, int c2, int width, int height, float h[3][3]) {
	int i, j, k, l;
	float value = 0;
	for (i = c1-1, k = 0 ; i <= c1+1 ; i++, k++){
		for (j = c2-1, l = 0 ; j <= c2+1 ; j++, l++){
			value += source[width * i + j] * h[k][l];
		}
	}
	destination[width * c1 + c2] = value;
}

void convoluteRgb(uint8_t *source, uint8_t *destination, int c1, int c2, int width, int height, float h[3][3]) {
	int i, j, k, l;
	float red = 0, green = 0, blue = 0;
	for (i = c1-1, k = 0 ; i <= c1+1 ; i++, k++){
		for (j = c2-3, l = 0 ; j <= c2+3 ; j+=3, l++){
			red += source[width * i + j]* h[k][l];
			green += source[width * i + j+1] * h[k][l];
			blue += source[width * i + j+2] * h[k][l];
		}
	}
	destination[width * c1 + c2] = red;
	destination[width * c1 + c2+1] = green;
	destination[width * c1 + c2+2] = blue;
}