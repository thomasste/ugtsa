#pragma once

#include <mutex>
#include "ugtsa/computation_graphs/computation_graph/computation_graph.h"

namespace computation_graphs {
namespace synchronized_computation_graph {

class Barrier {
public:
    explicit Barrier(std::size_t iCount) :
      mThreshold(iCount),
      mCount(iCount),
      mGeneration(0) {
    }

    void wait() {
        std::unique_lock<std::mutex> lLock{mMutex};
        auto lGen = mGeneration;
        if (!--mCount) {
            mGeneration++;
            mCount = mThreshold;
            mCond.notify_all();
        } else {
            mCond.wait(lLock, [this, lGen] { return lGen != mGeneration; });
        }
    }

private:
    std::mutex mMutex;
    std::condition_variable mCond;
    std::size_t mThreshold;
    std::size_t mCount;
    std::size_t mGeneration;
};

class ComputationGraph : public computation_graphs::computation_graph::ComputationGraph {
    computation_graphs::computation_graph::ComputationGraph *computation_graph;
    int thread_count;

    int waiting_threads;
    std::mutex mutex;
    Barrier barrier;

public:
    ComputationGraph(computation_graphs::computation_graph::ComputationGraph *computation_graph, int thread_count);

    int transformation(
        std::string seed,
        int seed_size,
        std::vector<std::string> inputs,
        std::vector<std::vector<int>> input_shapes,
        std::vector<tensorflow::DataType> input_types,
        std::vector<std::string> input_gradients,
        std::string output,
        std::vector<int> output_shape,
        tensorflow::DataType output_type,
        std::string output_gradient,
        std::string update_model_gradient_accumulators);
    int matrix(Eigen::VectorXi vector);
    int matrix(Eigen::VectorXf vector);
    int matrix(Eigen::MatrixXi matrix);
    int matrix(Eigen::MatrixXf matrix);
    int transformation_run(int transformation, std::vector<std::vector<int>> inputs);
    void run_batch();
    void accumulate_model_gradients(int first_node, std::vector<int> y_grad_indices, std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> y_grad_values);
    Eigen::VectorXf value(int index);
};

}
}