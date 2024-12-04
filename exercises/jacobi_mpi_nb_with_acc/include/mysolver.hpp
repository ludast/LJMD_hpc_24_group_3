#pragma once
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

  // To handle non-blocking communication
  MPI_Request request_send_up;
  MPI_Request request_recv_up;

  MPI_Request request_send_dw;
  MPI_Request request_recv_dw;

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

#pragma acc data copyin(field_ptr[0:N]) create(new_field_ptr[0:N])
  {
  for ( size_t steps = 0; steps < max_steps; steps++ ) {

    // TODO Specialize template for vector slicing

    // Sending and Receiving conditions
    // Rank 0: send my (my_height - 2) row to the (first) row of (rank + 1)
    // Rank last: send my 2nd row to the (last) row of (rank - 1)
    // Else: do both of the above cases
    // TIP: use MPI_PROC_NULL for dummy ranks


    /// Computation for the first bulk row
#pragma acc parallel loop
      for ( size_t j = 1; j < m.width - 1; j++ ) {
        new_field_ptr[1*(m.width) + j] = 0.25 * (
            field_ptr[1*(m.width) + j + 1] +
            field_ptr[1*(m.width) + j - 1] +
            field_ptr[1*(m.width) + j + (m.width)] +
            field_ptr[1*(m.width) + j - (m.width)]
            );
      }

//#pragma acc update self(new_field_ptr[0:N])

    //----------------------------------------
    // Send my 2nd row to prev_rank's last row
#pragma acc host_data use_device(field_ptr)
    MPI_Isend(&field_ptr[m.width], m.width, MPI_DOUBLE, prev_rank, 0,
        MPI_COMM_WORLD, &request_send_up);


    /// Computation for the last bulk row
#pragma acc parallel loop
      for ( size_t j = 1; j < m.width - 1; j++ ) {
        new_field_ptr[(m.my_height - 2)*(m.width) + j] = 0.25 * (
            field_ptr[(m.my_height - 2)*(m.width) + j + 1] +
            field_ptr[(m.my_height - 2)*(m.width) + j - 1] +
            field_ptr[(m.my_height - 2)*(m.width) + j + (m.width)] +
            field_ptr[(m.my_height - 2)*(m.width) + j - (m.width)]
            );
      }

//#pragma acc update self(new_field_ptr[0:N])

    //----------------------------------------
    // Send my 2nd last row to next_rank's first row
#pragma acc host_data use_device(field_ptr)
    MPI_Isend(&field_ptr[m.field.size() - 2*m.width], m.width, MPI_DOUBLE,
        next_rank, 0, MPI_COMM_WORLD, &request_send_dw);

    //----------------------------------------
    // Recv in my last row from next_rank's 2nd row
#pragma acc host_data use_device(field_ptr)
    MPI_Irecv(&field_ptr[m.field.size() - m.width], m.width, MPI_DOUBLE,
        next_rank, 0, MPI_COMM_WORLD, &request_recv_dw);

    //----------------------------------------
    // Recv in my first row from prev_rank's 2nd last row
#pragma acc host_data use_device(field_ptr)
    MPI_Irecv(&field_ptr[0], m.width, MPI_DOUBLE,
        prev_rank, 0, MPI_COMM_WORLD, &request_recv_up);


    /// Jacobi method
    /// At each iteration (step), the value of each inner matrix element
    /// needs to be recomputed from elements of the current iteration. The
    /// updating formula, based on numerical computation of second
    /// derivatives, is:

    /// Computation for bulk
#pragma acc parallel loop collapse(2)
      for ( size_t i = 2; i < m.my_height - 2; i++) {
        for ( size_t j = 1; j < m.width - 1; j++ ) {
          new_field_ptr[i*(m.width) + j] = 0.25 * (
              field_ptr[i*(m.width) + j + 1] +
              field_ptr[i*(m.width) + j - 1] +
              field_ptr[i*(m.width) + j + (m.width)] +
              field_ptr[i*(m.width) + j - (m.width)]
              );
        }
      }

    // Wait for non-blocking receive to complete
    MPI_Wait(&request_recv_up, MPI_STATUS_IGNORE);
    MPI_Wait(&request_recv_dw, MPI_STATUS_IGNORE);

    // The sender should not modify any part of the send buffer after a
    // nonblocking send operation is called, until the send completes.
    MPI_Wait(&request_send_up, MPI_STATUS_IGNORE);
    MPI_Wait(&request_send_dw, MPI_STATUS_IGNORE);

    /// Swap field <- new_field
    //m.field.swap(m.new_field);
    //std::swap(field_ptr, new_field_ptr);
#pragma acc parallel loop collapse(2)
      for ( size_t i = 1; i < m.my_height - 1; ++i) {
        for ( size_t j = 1; j < m.width - 1; ++j ) {
          field_ptr[i*(m.width) + j] = new_field_ptr[i*(m.width) + j];
        }
      }

//#pragma acc parallel num_gangs(1) num_workers(1) vector_length(1) present(field_ptr[:N], new_field_ptr[:N])
    //std::swap(field_ptr, new_field_ptr);

#ifdef PRINT
    if ( (steps % PrintInterval) == 0 ) {
#pragma acc update self(field_ptr[:N])
      m.print(steps);
    } // Print field
#endif

  }//END of iteration steps
  }//OpenACC Data

  endtime = MPI_Wtime(); // <---- END TIMER: runtime
  runtime = endtime - starttime;

  if (!rank){
    std::cout << runtime << std::endl;
  }

}
