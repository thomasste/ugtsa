#pragma once

#include "tensorflow/core/public/session.h"

#include "third_party/eigen3/Eigen/Core"

#include <vector>
#include <string>

namespace computation_graphs {
namespace computation_graph {

class ComputationGraph {
public:
    virtual int transformation(
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
        std::string update_model_gradient_accumulators) = 0;
    virtual int matrix(Eigen::VectorXi vector) = 0;
    virtual int matrix(Eigen::VectorXf vector) = 0;
    virtual int matrix(Eigen::MatrixXi matrix) = 0;
    virtual int matrix(Eigen::MatrixXf matrix) = 0;
    virtual int transformation_run(int transformation, std::vector<std::vector<int>> inputs) = 0;
    virtual void run_batch() = 0;
    virtual void accumulate_model_gradients(int first_node, std::vector<std::pair<int, Eigen::VectorXf>, Eigen::aligned_allocator<std::pair<int, Eigen::VectorXf>>> y_grads) = 0;
    virtual Eigen::VectorXf value(int index) = 0;
};

}
}