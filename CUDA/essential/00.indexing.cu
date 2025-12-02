#include <stdio.h>

__global__ void whoami(void) {

  /**
  * This formula Map each block in the grid with an index between 0 and number_of_blocks - 1
  * number_of_blocks represents the number of blocks in a grid
  */
  int block_id =                          // HOW TO KNOW apart_number infront of a building just knowing its coordinates???  Answer: count the building Dimension (GridDim) and do the math
    blockIdx.x +                          // apartment number on this floor (points across)
    blockIdx.y * gridDim.x +              // floor number in this building (rows high)
    blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep)

  
  int block_offset =
    block_id *                            // times our apartment number
    blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)
  
  /**
  * This formula Map each thread in the block with an index between 0 and number_of_threads - 1
  * number_of_threads represents the number of thread in a block
  */
  int thread_offset =
    threadIdx.x +
    threadIdx.y * blockDim.x +
    threadIdx.z * blockDim.x * blockDim.y;
    
  /**
  * This formula Map each thread in the grid with an incremental index starting from 0
  * It's literally the thread_id when with look at the grid without considering block segmentation 
  */
  int id = block_offset + thread_offset; // global person id in the entire apartment complex



  printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
    id,
    blockIdx.x, blockIdx.y, blockIdx.z, block_id,
    threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);

  // printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  
  // printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d  ---  blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d  ---  gridDim.x: %d, gridDim.y: %d, gridDim.z: %d, blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n",
  //   id,
  //   blockIdx.x, blockIdx.y, blockIdx.z, block_id,
  //   threadIdx.x, threadIdx.y, threadIdx.z, thread_offset,
  //   blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
  //   gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
}


int main(int argc, char** argv) {
  const int b_x = 2, b_y = 3, b_z = 4;
  const int t_x = 4, t_y = 4, t_z = 4; // the max warp size is 32, so 
  // we will get 2 warp of 32 threads per block

  int blocks_per_grid = b_x * b_y * b_z;
  int threads_per_block = t_x * t_y * t_z;

  printf("%d blocks/grid\n", blocks_per_grid);
  printf("%d threads/block\n", threads_per_block);
  printf("%d warps/block\n", threads_per_block / 32);
  printf("%d total threads\n", blocks_per_grid * threads_per_block);

  dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube of shape 2*3*4 = 24
  dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube of shape 4*4*4 = 64

  whoami<<<blocksPerGrid, threadsPerBlock>>>();
  // whoami<<<blocksPerGrid, 6>>>();
  cudaDeviceSynchronize();
}
