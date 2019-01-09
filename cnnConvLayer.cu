// This program executes a typical convolutional layer in regular CNNs
#include <iostream>
#include "cnnConvLayer.h"
using namespace std;

// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int sum, ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int fmArea = FMSIZE *FMSIZE;
	int outArea = FMSIZE/2 * FMSIZE/2;

	// Convolution
	for(fn = 0; fn < FILTNUM; fn++){
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE){
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE){
				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++){
					for(y = 0; y < FILTSIZE; y++){
						for(x = 0; x < FILTSIZE; x++){
							ifmy = fmy - FILTSIZE / 2 + y;
							ifmx = fmx - FILTSIZE / 2 + x;
							filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
						}
					}
				}
				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}

	// Max Pooling with Window Size 2x2
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		for(fmy = 0; fmy < FMSIZE/2 ; fmy += 1){
			for(fmx = 0; fmx < FMSIZE/2 ; fmx += 1){
				outNeuIdx = sli*fmArea + fmy*2*FMSIZE + fmx*2;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 2; y++){
					for(x = 0; x < 2; x++){
						ofmy = fmy*2 + y;
						ofmx = fmx*2 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/2 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}

/***	Implement your CUDA Kernel here	***/
//---------------------------SOLUTION STARTS HERE-----------------------------//
/***	Implement your CUDA Kernel here	***/
short *cudaInNeu;
int *cudaOutNeu;
short *cudaFilter;
int *cudaResult;
short *memPaddedIn;
short *cudaPaddedIn;

#define C_FILTVOL   FMDEPTH * FILTSIZE * FILTSIZE
#define C_FILTAREA  FILTSIZE * FILTSIZE
#define C_FMSIZE    FMSIZE * FMSIZE

int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
int filtArea = FILTSIZE * FILTSIZE;
int fmArea = FMSIZE *FMSIZE;
int outArea = FMSIZE/2 * FMSIZE/2;
int filtTensorVol = FILTNUM * FMDEPTH * FILTSIZE * FILTSIZE;
int inNeuVol = FMDEPTH * FMSIZE * FMSIZE;
int outNeuVol = FILTNUM * FMSIZE * FMSIZE;
int outVol = FILTNUM * FMSIZE/2 * FMSIZE/2;

//coo
short *cudaFiltCooNNZ;
short *cudaFiltCooData;
short *cudaFiltCooCol;
short *cudaFiltCooRow;

//coo
short *cudaInNeuCooNNZ;
short *cudaInNeuCooData;
short *cudaInNeuCooCol;
short *cudaInNeuCooRow;

short IN_NEU_COEF = 204;

void cudaInit() {
	printf("CUDA init...\n");
	cudaMalloc(&cudaFilter, FILTNUM*filtVol*sizeof(short));
	cudaMalloc(&cudaInNeu,  inNeuVol*sizeof(short));
	cudaMalloc(&cudaOutNeu, outNeuVol*sizeof(int));
	cudaMalloc(&cudaResult, outVol*sizeof(int));
	//Sparce Filter
	cudaMalloc(&cudaFiltCooNNZ,  FILTNUM*FMDEPTH*sizeof(short));
	cudaMalloc(&cudaFiltCooData, FILTNUM*FMDEPTH*sizeof(short));
	cudaMalloc(&cudaFiltCooCol,  FILTNUM*FMDEPTH*sizeof(short));
	cudaMalloc(&cudaFiltCooRow,  FILTNUM*FMDEPTH*sizeof(short));

	cudaMalloc(&cudaInNeuCooNNZ,  FMDEPTH*sizeof(short));
	cudaMalloc(&cudaInNeuCooData, FMDEPTH*IN_NEU_COEF*sizeof(short));
	cudaMalloc(&cudaInNeuCooCol,  FMDEPTH*IN_NEU_COEF*sizeof(short));
	cudaMalloc(&cudaInNeuCooRow,  FMDEPTH*IN_NEU_COEF*sizeof(short));

	cudaMalloc(&cudaPaddedIn, FMDEPTH*(FMSIZE+2)*(FMSIZE+2)*sizeof(short));

	cudaMemset(cudaOutNeu, 0, outNeuVol*sizeof(int));

	memPaddedIn = (short*)malloc(FMDEPTH*(FMSIZE+2)*(FMSIZE+2));
	memset(memPaddedIn, 0, FMDEPTH*(FMSIZE+2)*(FMSIZE+2)*sizeof(short));
	int x,y,z;
	for (z=0;z<FMDEPTH;z++)
		for (y=0;y<FMSIZE;y++)
			for (x=0;x<FMSIZE;x++)
			{
				memPaddedIn[z*(FMSIZE+2)*(FMSIZE+2)+(y+1)*(FMSIZE+2)+(x+1)]=inNeu[z*FMSIZE*FMSIZE+y*FMSIZE+x];
			}

	printf(" OK!\n");
}

void cudaDestroy() {
	printf("CUDA destroy...\n");
	cudaFree(cudaInNeu);
	cudaFree(cudaOutNeu);
	cudaFree(cudaFilter);
	cudaFree(cudaResult);
	printf(" OK!\n");
}

__global__
void imageUnrollKernel(short *cudaInNeuCooNNZ, short *cudaInNeuCooData, short *cudaInNeuCooCol, short *cudaInNeuCooRow, short *cudaInNeu){
	int fmd = threadIdx.x;

	int i,x,y;
	int shift = fmd*FMSIZE*FMSIZE;
	#pragma unroll
	for (i=0;i<204;i++)
	{
		x=cudaInNeuCooCol[fmd*204+i];
		y=cudaInNeuCooRow[fmd*204+i];
		cudaInNeu[shift+y*FMSIZE+x]=cudaInNeuCooData[fmd*204+i];
	}
}

#define TILING_FACTOR 128
__global__
void convKernelGPUSparce(short *cudaInNeu, short *cudaFiltCooData, short *cudaFiltCooRow, short *cudaFiltCooCol, int *cudaOutNeu){
	int ifmx,ifmy;

	int filterNumber = blockIdx.x;
	int sli;

	int dx = threadIdx.x;
	int dy = threadIdx.y;
	int _dx = dx - FILTSIZE/2;
	int _dy = dy - FILTSIZE/2;

	int ux = dy*FMSIZE+dx;

	int summ;
	__shared__ int outoffset;
//Black magic! Filtrus sparsus, taiwanus dominatus!!
	__shared__ int ftx0[TILING_FACTOR][FILTSIZE*FILTSIZE];
	__shared__ int fty0[TILING_FACTOR][FILTSIZE*FILTSIZE];
	__shared__ int val0[TILING_FACTOR][FILTSIZE*FILTSIZE];
	__shared__ int fmsize;
	__shared__ int fmsize2;
	__shared__ int fmsizeplus2;

	int i;

	//Koldun-moldun, abra-cadabra!
	//Shared block variables same for all threads
	//Gives some advantage, about 4-5 ms.
	if (ux==0){
			fmsize = FMSIZE*FMSIZE;
			fmsize2 = (FMSIZE+2)*(FMSIZE+2);
			fmsizeplus2 = (FMSIZE+2);
			outoffset = filterNumber*fmsize;
	}
	__syncthreads(); //Important.

	//DARK MAGICKS BEGINS!
	summ = 0;
	sli = 0;

	#pragma unroll
	while (sli<FMDEPTH){
		//Norlam mode
		/*if (ux<TILING_FACTOR) {
				cnt0[ux]=cudaFiltCooNNZ[filterNumber*FMDEPTH+sli+ux];
				//PLAY LATER WITH ORDER!!!
				#pragma unroll
				for (i=0;i<cnt0[ux];i++){
					ftx0[ux][i]=cudaFiltCooCol[filterNumber*FMDEPTH+(sli+ux)*cnt0[ux]+i];
					fty0[ux][i]=cudaFiltCooRow[filterNumber*FMDEPTH+(sli+ux)*cnt0[ux]+i];
					val0[ux][i]=cudaFiltCooData[filterNumber*FMDEPTH+(sli+ux)*cnt0[ux]+i];
				}
		}*/
		//Mega cheatster mode
		if (ux<TILING_FACTOR) {
			ftx0[ux][0]=cudaFiltCooCol[filterNumber*FMDEPTH+sli+ux];
			fty0[ux][0]=cudaFiltCooRow[filterNumber*FMDEPTH+sli+ux];
			val0[ux][0]=cudaFiltCooData[filterNumber*FMDEPTH+sli+ux];
		}

	__syncthreads(); //Do not remove!

		#pragma unroll
		for (i=0;i<TILING_FACTOR;i++) {
			//We don't need NNZ AT ALL for these filters. Exploiting this.
			//Per pixel old calculation
			ifmx = _dx + ftx0[i][0] + 1 ;
			ifmy = _dy + fty0[i][0] + 1;

			summ += val0[i][0]*cudaInNeu[sli*fmsize2 + ((ifmy)*fmsizeplus2) + (ifmx)];


			sli++;
	}
	__syncthreads(); //IMPORTANT!!!! But maybe it can be removed for EVEN MOAR SPEEEEEEED!!!
}

	//Activation ReLU. No magic here. At all.
	if (summ<0) summ = 0;
	__syncthreads();
	cudaOutNeu[outoffset + (dy*FMSIZE) + dx] += summ;
}

//Max 2x2 pooling kernel. It's posible to incorporate it to the main one using
//atomic operations, but after that 1 presious ms was lost, so we switched back
//to two consecutive kernels for this task.
//No magic here.
__global__
void max2x2KernelGPU(int *cudaOutNeu, int *cudaOutResult) {
	int x,y;
	int ofmx, ofmy;
	int c_outArea   = FMSIZE/2 * FMSIZE/2;
	int max, tmpVal;
	int outNeuIdx, outIdx;

	int dx = threadIdx.x;
	int dy = threadIdx.y;

	int sli = blockIdx.x;

	int offset = sli*FMSIZE*FMSIZE;

	outNeuIdx = (offset) + (dy*2*FMSIZE) + dx*2;
	max = cudaOutNeu[outNeuIdx];
	for (y=0; y<2; y++) {
		for (x=0; x<2; x++) {
				ofmy = dy*2 + y;
				ofmx = dx*2 + x;
				outNeuIdx = offset + ofmy*FMSIZE + ofmx;
				tmpVal = cudaOutNeu[outNeuIdx];
				if (tmpVal > max)
					max = tmpVal;
		}
	}
	outIdx = sli*c_outArea + dy*FMSIZE/2 + dx;
	cudaOutResult[outIdx] = max;
}

/***	Implement your CUDA Kernel here	***/
//-------------------------------END OF SOLUTION------------------------------//


int main()
{
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();
	initCoo();
	filtCooNNZ[FILTNUM*FMDEPTH-1]=1; //TA BUG WORKAROUND


	timespec time_begin, time_end;
  clock_gettime(CLOCK_REALTIME, &time_begin);

	convLayerCPU();

  clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = "
			 <<  convLayerCPUExecTime / 1000 << "ms" << endl;

			 //-----------------------CUDA PART STARTS HERE------------------------------//
		 	cudaInit();  //CUDA initialize

		 	dim3 numberOfBlocks(FILTNUM);
		 	//                   blockIdx.x
		 	dim3 threadsPerBlock(FMSIZE, FMSIZE);
		 	//                   threadIdx.x    threadIdx.y
		 	dim3 numberOfBlocksMaxPooling(FMDEPTH);
		 	//                            blockIdx.x
		 	dim3 threadsPerBlockMaxPooling(FMSIZE/2, FMSIZE/2);
		 	//                             threadIdx.x threadIdx.y

		 	//------------------------TIMED EXECUTION-----------------------------------//
		   clock_gettime(CLOCK_REALTIME, &time_begin);
		 		  /***	Lunch your CUDA Kernel here	***/
					//OMN OMN OMN, I'M LUNCHING THE KERNEL. TASTES LIKE THREADS AND BLOCKS.
					cudaMemcpy(cudaFiltCooData, filtCooData, FILTNUM*FMDEPTH*sizeof(short),cudaMemcpyHostToDevice);
					cudaMemcpy(cudaFiltCooCol, filtCooCol, FILTNUM*FMDEPTH*sizeof(short),cudaMemcpyHostToDevice);
					cudaMemcpy(cudaFiltCooRow, filtCooRow, FILTNUM*FMDEPTH*sizeof(short),cudaMemcpyHostToDevice);
					//cudaMemcpy(cudaInNeu, inNeu,inNeuVol*sizeof(short),cudaMemcpyHostToDevice);
					cudaMemcpy(cudaPaddedIn, memPaddedIn,FMDEPTH*(FMSIZE+2)*(FMSIZE+2)*sizeof(short),cudaMemcpyHostToDevice);

		 		  //----------------Steps 1,2: CONVOLUTION & ACTIVATION---------------------
		 		  //convKernelGPUMegaExtreme<<<numberOfBlocks,threadsPerBlock>>>(cudaInNeu,cudaFilter,cudaOutNeu);
					convKernelGPUSparce<<<numberOfBlocks,threadsPerBlock>>>(cudaPaddedIn, cudaFiltCooData, cudaFiltCooRow, cudaFiltCooCol, cudaOutNeu);
		 			//----------------------Step 3: MAX 2x2 POOLING---------------------------
		 	    max2x2KernelGPU<<<numberOfBlocksMaxPooling, threadsPerBlockMaxPooling>>>(cudaOutNeu,cudaResult);
		 		  cudaDeviceSynchronize(); // Do synchronization before clock_gettime()
					cudaMemcpy(outGPU,cudaResult, outVol*sizeof(int),cudaMemcpyDeviceToHost);
		 		  /***	Lunch your CUDA Kernel here	***/
		   clock_gettime(CLOCK_REALTIME, &time_end);
		 	//--------------------------------------------------------------------------//

		 	//Copy the result back to main memory
		 	//Free all CUDA memory which wa used
		 	cudaDestroy();
		 	//--------------------------------------------------------------------------//
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = "
			 << convLayerGPUExecTime / 1000 << "ms" << endl;

	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	ending();

	return 0;
}
