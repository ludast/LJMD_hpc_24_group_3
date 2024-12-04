#include <mpi.h>
#include <openacc.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "mpi_comm.hpp"
#include "mymesh.hpp"
#include "mysolver.hpp"


////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  ///-----------------------------------
  int N;                /// size of the grid
  double corner;        /// value of the corner
  double i_values;      /// initial values of the field (0.5)
  int max_steps;
  int PrintInterval;

  /// Read parameters
  std::ifstream paramvar("params.conf");
  paramvar >> N >> corner >> i_values >> max_steps >> PrintInterval;
  paramvar.close();
  ///-----------------------------------

  // We require at most N processes
  if (world_size > N) {
    if ( !rank ) {
      std::cerr << "World size must not be greater than " << N << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ///-----------------------------------
  /// Create the grid
  CMesh<double> grid(N, boundary_conditions<double>,
      corner, i_values,
      rank, world_size, MPI_COMM_WORLD);

  // int step{0};
  // grid.print_wg(step);
  ///-----------------------------------

  ///-----------------------------------
  /// Solve it
  CSolver<double> solve;
  solve.jacobi(grid, max_steps, PrintInterval );
  ///-----------------------------------

  MPI_Finalize();
  return 0;

}
