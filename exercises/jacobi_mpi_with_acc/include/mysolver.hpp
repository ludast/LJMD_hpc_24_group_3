#pragma once
#include <cstring>
////////////////////////////////////////////////////////////////////////////////
/// A solver class with the jacobi method
template <typename T>
class CSolver{
  public:
    void jacobi(CMesh<T>& m, const size_t& max_steps, const size_t& PrintInterval);
};

/// Jacobi Method
template <typename T>
void CSolver<T>::jacobi( CMesh<T>& m, const size_t& max_steps, const size_t& PrintInterval ) {

  auto rank = m.rank;
  auto world_size = m.world_size;

  // Example: Send to the next process, receive from the previous process
  //without "periodic boundary conditions"
  int next_rank = (rank == world_size - 1) ? MPI_PROC_NULL : rank + 1;
  int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;

  // OpenACC pointers
  size_t N = m.field.size();
  auto* __restrict__ field_ptr = m.field.data();
  auto* __restrict__ new_field_ptr = m.new_field.data();

  // Timing variables
  double starttime, endtime, runtime;
  starttime = MPI_Wtime(); // START TIMER: runtime

  for ( size_t steps = 0; steps < max_steps; steps++ ) {

    // TODO Specialize template for vector slicing
    // Sending and Receiving conditions
    // Rank 0: send my (my_height - 2) row to the (first) row of (rank + 1)
    // Rank last: send my 2nd row to the (last) row of (rank - 1)
    // Else: do both of the above cases
    // TIP: use MPI_PROC_NULL for dummy ranks


    // send (2nd last row) to (first)
    MPI_Sendrecv(&m.field[m.field.size() - 2*m.width], m.width,
        MPI_DOUBLE, next_rank, 0, &m.field[0], m.width, MPI_DOUBLE,
        prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // send 2nd row to (last)
    MPI_Sendrecv(&m.field[m.width], m.width, MPI_DOUBLE, prev_rank, 0,
        &m.field[m.field.size() - m.width], m.width, MPI_DOUBLE,
        next_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    /// Jacobi method
#pragma acc parallel loop independent copyin(field_ptr[:N]) copy(new_field_ptr[:N])
    for ( size_t i = 1; i < m.my_height - 1; i++) {
#pragma acc loop
      for ( size_t j = 1; j < m.width - 1; j++ ) {
        new_field_ptr[i*(m.width) + j] = 0.25 * (
            field_ptr[i*(m.width) + j + 1] +
            field_ptr[i*(m.width) + j - 1] +
            field_ptr[i*(m.width) + j + (m.width)] +
            field_ptr[i*(m.width) + j - (m.width)]
            );
      }
    }

    /// Swap field <- new_field
    //std::swap(new_field_ptr, field_ptr);
    std::memcpy(&field_ptr[0], &new_field_ptr[0], N*sizeof(T));
    //m.field.swap(m.new_field);
//#pragma acc parallel loop independent //collapse(2)
      //for ( size_t i = 1; i < m.my_height - 1; ++i) {
//#pragma acc loop
        //for ( size_t j = 1; j < m.width - 1; ++j ) {
          //field_ptr[i*(m.width) + j] = new_field_ptr[i*(m.width) + j];
        //}
      //}

#ifdef PRINT
    if ( (steps % PrintInterval) == 0 ) {
//#pragma acc update self(field_ptr[:N])
      m.print(steps);
    } // Print field
#endif

  }//END of steps iteration

  endtime = MPI_Wtime();
  runtime = endtime - starttime;

  std::cout << runtime << std::endl;

}
