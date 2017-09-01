#pragma once

#include "tensorflow/core/public/session.h"

#include "third_party/eigen3/Eigen/Core"

#include <vector>
#include <string>

namespace computation_graphs {
namespace computation_graph {

class ComputationGraph {
protected:
    bool training = false;

public:
    ComputationGraph(bool training);

    virtual int transformation(
        std::string training,
        std::string seed,
        int seed_size,
        std::vector<std::string> inputs,
        std::vector<std::vector<int>> input_shapes,
        std::vector<tensorflow::DataType> input_types,
        std::vector<std::string> input_gradients,
        std::string output,
        std::vector<int> output_shape,
        tensorflow::DataType output_type,
        std::string output_gradient) = 0;
    virtual int matrix(Eigen::VectorXi vector) = 0;
    virtual int matrix(Eigen::VectorXf vector) = 0;
    virtual int matrix(Eigen::MatrixXi matrix) = 0;
    virtual int matrix(Eigen::MatrixXf matrix) = 0;
    virtual int transformation_run(int transformation, std::vector<std::vector<int>> inputs) = 0;
    virtual void run_batch() = 0;
    virtual Eigen::VectorXf value(int index) = 0;
};

}
}