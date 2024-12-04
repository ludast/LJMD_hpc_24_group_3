#pragma once
////////////////////////////////////////////////////////////////////////////////
/// Mesh class to store the grid
template <typename T>
class CMesh {
  public:

    /// Size of the mesh
    size_t width;
    /// Old field
    std::vector<T> field;
    /// New field
    std::vector<T> new_field;

    /// mesh variables
    T corner;
    T i_values;

    /// MPI variables
    int rank;
    int world_size;
    MPI_Comm comm;
    MPI_Status status;

    /// MPI local variables
    size_t N_loc{0};
    size_t my_height{0};
    int res;
    int offset;

    // buffer to receive data to print
    std::vector<T> buff;
    size_t h_size{0};

    /// Constructor
    /// Theh constructor should take the size as a parameter and also the
    /// function for setting up the boundary conditions.
    template <typename F>
      CMesh( const size_t& N, const F& boundary_conditions,
          const T& corner, const T& i_values,
          const int& rank_entry,
          const int& world_size_entry,
          const MPI_Comm& comm_entry
          );

    /// Printing function
    void print( const int& step );

    /// Printing function with ghost cells
    void print_wg( const int& step );

};


/// Constructor
/// initialize the mesh
template <typename T>
template <typename F>
CMesh<T>::CMesh( const size_t& N, const F& boundary_conditions,
    const T& corner_entry, const T& values_entry,
    const int& rank_entry,
    const int& world_size_entry,
    const MPI_Comm& comm_entry
    ) {

  width  = N + 2;

  corner = corner_entry;
  i_values = values_entry;

  // MPI variables
  rank = rank_entry;
  world_size = world_size_entry;
  comm = comm_entry;

  // number of rows will be divided equally and the rest in a round-robin way
  N_loc = N / world_size;
  res = N % world_size;
  my_height = rank < res ? N_loc + 2 + 1 : N_loc + 2;
  offset    = rank < res ? 0 : res ;

  // resize my field and my new_field
  field.resize( (my_height)*(width), i_values );
  new_field.resize( (my_height)*(width), i_values );


  //You need to leave space for the boundaries, as "by logic" they do not
  //belong to the main grid (i.e. a 1024 x 1024 matrix would need to actually
  //be 1026x1026).

  boundary_conditions(field, width, my_height, offset, rank, world_size, corner, i_values);
  boundary_conditions(new_field, width, my_height, offset, rank, world_size, corner, i_values);

}

/// Boundary Conditions
template < typename T >
void boundary_conditions( std::vector<T>& field,
    const size_t& width,
    const size_t& my_height,
    const int offset,
    const int& rank,
    const int& world_size,
    const T& corner, const T& i_values ) {

  T delta;
  delta = corner / (width - 1);

  // compute the last left-border value of the previous process
  // rank x N_loc x delta + offset * delta // (add offset for the rest of PES)
  T h_offset = (rank * (my_height-2) + offset)* delta;

  // fill inner values here or in the constructor
  // std::fill(field.begin(), field.end(), i_values);

  // the first process fill the top row
  if ( !rank ) {
    for ( size_t i = 0; i < width; i++ ){
      field[i] = 0;                                     /// top row
    }
  }

  // the last process fill the bottom row
  if ( rank == (world_size - 1) ) {
    for ( size_t i = 0; i < width; i++ ){
      field[ ( my_height - 1 )*width + i ] = corner - i*delta; /// bottom row
    }
  }

  for ( size_t i = 0; i < my_height; i++ ) {
    field[ i*width ] = i*delta + h_offset;
    // field[ (width - 1 )*width + i ] = corner - i*delta; /// bottom row
    // field[i] = 0;                                     /// top row
    field[(i + 1)*width - 1] = 0;                      /// right-most column
  }

}

/// Printing function
template <typename T>
void CMesh<T>::print( const int& step ) {

  //Printing conditions
  // if I am rank 0: print from row 0 to (my_height - 1)
  // if I am the last rank: print from row 1 to (my_height)
  // else: print from row 1 to (my_height - 1)

  if ( !rank ) {

    std::ostringstream ss;
    ss << "./build/dat/mesh_" << std::setfill('0') << std::setw(4) << step << ".dat";
    std::ofstream filevar;
    std::string result = ss.str();

    filevar.open(result);

    // if I am the only one, print the complete field
    if ( world_size == 1 ) {
      // print my part of the matrix
      for ( size_t i = 0; i < (my_height); i++ ) {
        filevar << field[i*width];
        for ( size_t j = 1; j < width; j++ ) {
          filevar << " " << field[i*width + j];
        }
        filevar << std::endl;
      }
    }
    else {
      // print my part of the matrix
      for ( size_t i = 0; i < (my_height - 1); i++ ) {
          filevar << field[i*width];
        for ( size_t j = 1; j < width; j++ ) {
          // filevar << std::fixed << std::setprecision(4) << field[i*width + j] << " ";
          filevar << " " << field[i*width + j];
        }
        filevar << std::endl;
      }
    }

    for ( int i = 1; i < world_size; i++ ) {
      // receive the number of rows and resize buffer;
      mpi_recv(h_size, i, 0, comm);
      buff.resize(h_size*width);
      // Probe for an incoming message from workers
      mpi_recv_vec(buff, i, 0, comm);

      // print buffer;
      for ( size_t i = 0; i < h_size; i++ ) {
          filevar << buff[i*width];
        for ( size_t j = 1; j < width; j++ ) {
          filevar << " " << buff[i*width + j];
        }
        filevar << std::endl;
      }
    }

    filevar.close();
  } else if ( rank == (world_size - 1) ) {
    // send the number of rows first;
    mpi_send(my_height-1, 0, 0, comm);
    // TODO ask how to pass a sub vector without duplicating data
    mpi_send_vec(std::vector<T>(field.begin()+width, field.end()), 0, 0, comm);
  } else {
    // send the number of rows first;
    mpi_send(my_height-2, 0, 0, comm);
    mpi_send_vec(std::vector<T>(field.begin()+width, field.end()-width), 0, 0, comm);
  }


}


/// Printing function with ghost cells
template <typename T>
void CMesh<T>::print_wg( const int& step ) {


  if ( !rank ) {

    std::ostringstream ss;
    ss << "./build/dat/mesh_" << std::setfill('0') << std::setw(4) << step << ".dat";
    std::ofstream filevar;
    std::string result = ss.str();

    filevar.open(result);

    // print my part of the matrix
    for ( size_t i = 0; i < my_height; i++ ) {
      filevar << field[i*width];
      for ( size_t j = 1; j < width; j++ ) {
        filevar << " " << field[i*width + j];
      }
      filevar << std::endl;
    }
    // NOTE spacing
    filevar << std::endl;

    for ( int i = 1; i < world_size; i++ ) {
      // receive the number of rows and resize buffer;
      mpi_recv(h_size, i, 0, comm);
      buff.resize(h_size*width);
      // Probe for an incoming message from workers
      mpi_recv_vec(buff, i, 0, comm);

      // print buffer;
      for ( size_t i = 0; i < h_size; i++ ) {
          filevar << buff[i*width];
        for ( size_t j = 1; j < width; j++ ) {
          filevar << " " << buff[i*width + j];
        }
        filevar << std::endl;
      }
      // NOTE spacing
      filevar << std::endl;
    }

    filevar.close();
  } else {
    // send the number of rows first;
    mpi_send(my_height, 0, 0, comm);
    mpi_send_vec(field, 0, 0, comm);
  }


}

