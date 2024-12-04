#pragma once

#include <map>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>

/// Class Timer Data
typedef struct TimerData {
  /// Number of calls
  int calls{0};
  /// time taken for every call
  long int tspan;
} TimerData;

// Each processor will populate its own table
std::map<int, std::map<int, TimerData>> ttt; // TaskTimeTable
std::map<int, std::map<int, TimerData>>::iterator i_ttt;
std::map<int, TimerData>::iterator j_ttt;

// Only Root will populate this table
std::map<int, std::map<std::string, int>> mmm; // Min Max Mean
std::map<int, std::map<std::string, int>>::iterator i_mmm;
std::map<std::string, int>::iterator j_mmm;

/// Class Simple Timer
class CSimple_Timer{
  public:
    //DECLARE VARIABLES FOR WHAT WE ARE TIMING, FOR CLOCK START AND END
    using time_units = std::chrono::milliseconds;
    std::chrono::time_point<std::chrono::steady_clock> t_start;
    std::chrono::time_point<std::chrono::steady_clock> t_end;
    int key;
    int rank; // MPI rank and root

    //constructor
    CSimple_Timer( const int& timewhat0, const int& myrank) {
      //SET WHAT WE ARE TIMING FROM THE PASSED PARAMETER
      key = timewhat0;
      rank = myrank;

      //LOOK FOR THE STRING OCCURENCE AND EITHER AUGMENT THE CALLS OR 
      //INSERT A NEW THING INTO THE TABLE
      ttt[key][rank].calls += 1;

      //START THE CLOCKS
      t_start = std::chrono::steady_clock::now();

    };

    //destructor
    ~CSimple_Timer(){
      //STOP THE CLOCK
      t_end = std::chrono::steady_clock::now();
      //CALCULATE DURATION
      auto tspan = std::chrono::duration_cast<time_units>(t_end - t_start).count();
      //INSERT THAT INTO YOUR TABLE  
      ttt[key][rank].tspan += tspan;
    }//destructor

    // static class functions - can be called without the object of the class
    static void print_timing_results(){

      for (i_ttt = ttt.begin(); i_ttt != ttt.end(); i_ttt++) {
        std::cout << i_ttt->first << " : ";
        for (j_ttt = i_ttt->second.begin();
            j_ttt != i_ttt->second.end(); j_ttt++) {
          std::cout << j_ttt->first << " ";
          std::cout << j_ttt->second.calls << " ";
          std::cout << j_ttt->second.tspan << " , ";
        }
        std::cout << std::endl;
      }

    };

     // JSON formatting
    static void print_timing_results_json(){

      for (i_ttt = ttt.begin(); i_ttt != ttt.end(); i_ttt++) {
        std::cout << "{ \"" << i_ttt->first << "\" : [ ";
        for (j_ttt = i_ttt->second.begin();
            j_ttt != i_ttt->second.end(); j_ttt++) {
          std::cout << "{\"" << j_ttt->first << "\" : " ;
          std::cout << "{ \"calls\" : " << j_ttt->second.calls << ", ";
          std::cout << "\"duration_ms\" : " << j_ttt->second.tspan << "} }, ";

          // TODO  remove trailing comma at the last entry ---------------^
        }
        std::cout << " ] }" << std::endl;
      }

    };

    // Print average, minimum and maximum values for each function measured
    static void print_summary(){

      std::cout << std::endl;
      for ( i_mmm = mmm.begin(); i_mmm != mmm.end(); i_mmm++ ) {
        std::cout << i_mmm->first << " :";
        for ( j_mmm = i_mmm->second.begin();
            j_mmm != i_mmm->second.end(); j_mmm++ ) {
          std::cout << " " << j_mmm->first << " " << j_mmm->second;
        }
        std::cout << std::endl;
      }
    };


    // Every one sends its table to the root process
    static void gather_times( const int& rank, const int& root, const int& world_size){
      // Each worker has its own TimeTable
      std::vector<int> send_keys;
      std::vector<int> send_myid;
      std::vector<int> send_calls;
      std::vector<int> send_times;

      // Lets split each field of the TimeTable
      for ( i_ttt = ttt.begin(); i_ttt != ttt.end(); i_ttt++ ) {
        send_keys.push_back(i_ttt->first);
        for ( j_ttt = i_ttt->second.begin(); j_ttt != i_ttt->second.end(); j_ttt++ ) {
          send_myid.push_back(j_ttt->first);
          send_calls.push_back(j_ttt->second.calls);
          send_times.push_back(j_ttt->second.tspan);
        }
      }

      // Vector for gathering tasks done by each worker (significant only at root)
      std::vector<int> recv_counts;
      std::vector<int> displs;
      int ni_ttts = send_keys.size(); // (narrowing conversion)
      int totaltasks;

      if ( rank == root ) {
        recv_counts.resize(world_size);
        displs.resize(world_size, 0);
      }

      // vector of tasks done by each worker
      MPI_Gather(&ni_ttts, 1, MPI_INT,
          recv_counts.data(), 1, MPI_INT, root, MPI_COMM_WORLD);
      // total number of tasks in world
      MPI_Reduce(&ni_ttts, &totaltasks, world_size, MPI_INT,
          MPI_SUM, root, MPI_COMM_WORLD);

      // Vectors for gathering information (significant only at root)
      std::vector<int> recv_keys;
      std::vector<int> recv_ids;
      std::vector<int> recv_calls;
      std::vector<int> recv_times;

      if ( rank == root ) {
        recv_keys.resize(totaltasks);
        recv_ids.resize(totaltasks);
        recv_calls.resize(totaltasks);
        recv_times.resize(totaltasks);

        // compute the displacement relative to recvbuf
        for ( size_t i = 1; i < recv_counts.size(); i++ ) {
          displs[i] = displs[i-1] + recv_counts[i-1];
        }

      }

      // Gather labels
      MPI_Gatherv(send_keys.data(), send_keys.size(), MPI_INT, recv_keys.data(),
          recv_counts.data(), displs.data(), MPI_INT, root, MPI_COMM_WORLD);
      // Gather proc_ids
      MPI_Gatherv(send_myid.data(), send_myid.size(), MPI_INT, recv_ids.data(),
          recv_counts.data(), displs.data(), MPI_INT, root, MPI_COMM_WORLD);
      // Gather calls
      MPI_Gatherv(send_calls.data(), send_calls.size(), MPI_INT, recv_calls.data(),
          recv_counts.data(), displs.data(), MPI_INT, root, MPI_COMM_WORLD);
      // Gather times
      MPI_Gatherv(send_times.data(), send_times.size(), MPI_INT, recv_times.data(),
          recv_counts.data(), displs.data(), MPI_INT, root, MPI_COMM_WORLD);

      if ( rank == root ) {

        // Reconstruct the Time Table
        for ( size_t i = 0; i < recv_keys.size(); i++ ) {
          ttt[ recv_keys[i] ][ recv_ids[i] ].calls = recv_calls[i];
          ttt[ recv_keys[i] ][ recv_ids[i] ].tspan = recv_times[i];
        }

        // Populate the Summary Table
        int tcum, tavg;

        for (i_ttt = ttt.begin(); i_ttt != ttt.end(); i_ttt++) {

          // search max
          j_ttt = std::max_element(i_ttt->second.begin(), i_ttt->second.end(), []
              (const std::pair<int,TimerData>& a, const std::pair<int,TimerData>&
               b)->bool{ return a.second.tspan < b.second.tspan; } );
          // push the maximum value to the summary table
          mmm[i_ttt->first]["max"] = j_ttt->second.tspan;

          // search min
          j_ttt = std::min_element(i_ttt->second.begin(), i_ttt->second.end(), []
              (const std::pair<int,TimerData>& a, const std::pair<int,TimerData>&
               b)->bool{ return a.second.tspan < b.second.tspan; } );
          // push the minimum value to the summary table
          mmm[i_ttt->first]["min"] = j_ttt->second.tspan;

          // accumulate time values
          tcum = std::accumulate(i_ttt->second.begin(), i_ttt->second.end(), 0, []
              (int acc, const std::pair<int,TimerData>& a)
              { return acc + a.second.tspan; } );

          // average of time values
          tavg = tcum / i_ttt->second.size();
          // push the time average to the summary table
          mmm[i_ttt->first]["avg"] = tavg;


        }
      }
    }

};
