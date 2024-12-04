#pragma once

// Specializing templates
template<typename T>
void mpi_send(const T& data, int dest, int tag, MPI_Comm comm) {
  if constexpr (std::is_same<T, int>::value) {
    MPI_Send(&data, 1, MPI_INT, dest, tag, comm);
  } else if constexpr (std::is_same<T, double>::value) {
    MPI_Send(&data, 1, MPI_DOUBLE, dest, tag, comm);
  } else {
    MPI_Send(&data, sizeof(T), MPI_BYTE, dest, tag, comm);
  }
}

template<typename T>
void mpi_recv(T& data, int source, int tag, MPI_Comm comm) {
  if constexpr (std::is_same<T, int>::value) {
    MPI_Recv(&data, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
  } else if constexpr (std::is_same<T, double>::value) {
    MPI_Recv(&data, 1, MPI_DOUBLE, source, tag, comm, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(&data, sizeof(T), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
  }
}


// Specializing templates for vector:
template<typename T>
void mpi_send_vec(const std::vector<T>& vec, int dest, int tag, MPI_Comm comm) {
  if constexpr (std::is_same<T, int>::value) {
    MPI_Send(vec.data(), vec.size(), MPI_INT, dest, tag, comm);
  } else if constexpr (std::is_same<T, double>::value) {
    MPI_Send(vec.data(), vec.size(), MPI_DOUBLE, dest, tag, comm);
  } else {
    MPI_Send(vec.data(), sizeof(T), MPI_BYTE, dest, tag, comm);
  }
}

template<typename T>
void mpi_recv_vec(std::vector<T>& vec, int source, int tag, MPI_Comm comm) {
  if constexpr (std::is_same<T, int>::value) {
    MPI_Recv(vec.data(), vec.size(), MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
  } else if constexpr (std::is_same<T, double>::value) {
    MPI_Recv(vec.data(), vec.size(), MPI_DOUBLE, source, tag, comm, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(vec.data(), sizeof(T), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
  }
}

