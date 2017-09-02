#include "ugtsa/computation_graphs/basic_computation_graph/computation_graph.h"

namespace computation_graphs {
namespace basic_computation_graph {

ComputationGraph::ComputationGraph(tensorflow::Session* session, std::string training_name, bool training)
        : computation_graphs::computation_graph::ComputationGraph(), session(session), training_name(training_name) {
    training_tensor = tensorflow::Tensor(tensorflow::DataType::DT_BOOL, tensorflow::TensorShape({}));
    training_tensor.scalar<bool>()(0) = training;

    batches.push_back({
        0,
        {}
    });
}

int ComputationGraph::transformation(
        std::string seed,
        int seed_size,
        std::vector<std::string> inputs,
        std::vector<std::vector<int>> input_shapes,
        std::vector<tensorflow::DataType> input_types,
        std::vector<std::string> input_gradients,
        std::string output,
        std::vector<int> output_shape,
        tensorflow::DataType output_type,
        std::string output_gradient) {
    std::vector<int> input_sizes;
    for (auto &input_shape : input_shapes) {
        int input_size = 1;
        for (int input_dim : input_shape) input_size *= input_dim;
        input_sizes.push_back(input_size);
    }
    int output_size = 1;
    for (int output_dim : output_shape) output_size *= output_dim;

    transformations.push_back({
        seed,
        seed_size,
        inputs,
        input_shapes,
        input_sizes,
        input_types,
        input_gradients,
        output,
        output_shape,
        output_size,
        output_type,
        output_gradient
    });

    return transformations.size() - 1;
}

int ComputationGraph::matrix(Eigen::VectorXi vector) {
    std::vector<int> output;
    for (int i = 0; i < vector.size(); i++) {
        output.push_back(vector(i));
    }

    nodes.push_back({
        -1,
        {},
        output
    });

    return nodes.size() - 1;
}

int ComputationGraph::matrix(Eigen::VectorXf vector) {
    std::vector<int> output;
    for (int i = 0; i < vector.size(); i++) {
        output.push_back(*((int *)&vector(i)));
    }

    nodes.push_back({
        -1,
        {},
        output
    });

    return nodes.size() - 1;
}

int ComputationGraph::matrix(Eigen::MatrixXi matrix) {
    std::vector<int> output;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            output.push_back(matrix(i, j));
        }
    }

    nodes.push_back({
        -1,
        {},
        output
    });

    return nodes.size() - 1;
}

int ComputationGraph::matrix(Eigen::MatrixXf matrix) {
    std::vector<int> output;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            output.push_back(*((int *)&matrix(i, j)));
        }
    }

    nodes.push_back({
        -1,
        {},
        output
    });

    return nodes.size() - 1;
}

int ComputationGraph::transformation_run(int transformation, std::vector<std::vector<int>> inputs) {
    nodes.push_back({
        transformation,
        inputs,
        {}
    });

    return nodes.size() - 1;
}

void ComputationGraph::run_batch() {
    // create seeds
    std::uniform_int_distribution<long long> distribution;
    std::vector<std::vector<long long>> seeds;

    for (auto &transformation : transformations) {
        std::vector<long long> seed;
        for (int i = 0; i < transformation.seed_size; i++) {
            seed.push_back(distribution(generator));
        }
        seeds.push_back(seed);
    }

    batches.push_back({
        (int) nodes.size(),
        seeds
    });

    // collect inputs
    std::vector<int> transformations_input_count;
    std::vector<std::vector<std::vector<int>>> transformations_input_buffers;

    for (auto &transformation : transformations) {
        std::vector<std::vector<int>> transformation_input_buffers;
        for (int i = 0; i < transformation.inputs.size(); i++) {
            transformation_input_buffers.push_back({});
        }
        transformations_input_count.push_back(0);
        transformations_input_buffers.push_back(transformation_input_buffers);
    }

    for(int node_index = batches[batches.size() - 2].nodes_end; node_index < batches[batches.size() - 1].nodes_end; node_index++) {
        auto &node = nodes[node_index];
        if (node.transformation != -1) {
            transformations_input_count[node.transformation] += 1;
            auto &transformation = transformations[node.transformation];
            auto &transformation_input_buffers = transformations_input_buffers[node.transformation];

            for (int input_index = 0; input_index < transformation.inputs.size(); input_index++) {
                auto &node_input = node.inputs[input_index];
                auto &transformation_input_buffer = transformation_input_buffers[input_index];
                auto &transformation_input_size = transformation.input_sizes[input_index];

                int partial_inputs_size = 0;

                for (auto partial_input_node_index : node_input) {
                    auto &partial_input = nodes[partial_input_node_index].output;
                    partial_inputs_size += partial_input.size();
                    transformation_input_buffer.insert(transformation_input_buffer.end(), partial_input.begin(), partial_input.end());
                }

                for (int i = 0; i < transformation_input_size - partial_inputs_size; i++) {
                    transformation_input_buffer.push_back(0);
                }
            }
        }
    }

    // create feed dict
    std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;

    for (int transformation_index = 0; transformation_index < transformations.size(); transformation_index++) {
        auto &transformation = transformations[transformation_index];
        int transformation_input_count = transformations_input_count[transformation_index];
        auto &transformation_input_buffers = transformations_input_buffers[transformation_index];
        if (transformation_input_count > 0) {
            for (int input_index = 0; input_index < transformation.inputs.size(); input_index++) {
                auto &transformation_input_buffer = transformation_input_buffers[input_index];
                auto &input_name = transformation.inputs[input_index];
                auto &input_shape = transformation.input_shapes[input_index];
                auto &input_type = transformation.input_types[input_index];
                tensorflow::TensorShape tensor_shape;
                tensor_shape.AddDim(transformation_input_count);
                for (int input_dim : input_shape) {
                    tensor_shape.AddDim(input_dim);
                }
                tensorflow::Tensor tensor(input_type, tensor_shape);
                if (input_type == tensorflow::DataType::DT_FLOAT) {
                    for (int i = 0; i < transformation_input_buffer.size(); i++) {
                        tensor.flat<float>()(i) = *((float *)&transformation_input_buffer[i]);
                    }
                } else if (input_type == tensorflow::DataType::DT_INT32) {
                    for (int i = 0; i < transformation_input_buffer.size(); i++) {
                        tensor.flat<int>()(i) = transformation_input_buffer[i];
                    }
                } else {
                    std::cout << "unsupported type: " << input_type << std::endl;
                    exit(1);
                }
                feed_dict.push_back(std::make_pair(input_name, tensor));
            }
        }
    }

    for (int transformation_index = 0; transformation_index < transformations.size(); transformation_index++) {
        auto &transformation = transformations[transformation_index];
        auto &input_name = transformation.seed;
        auto &seed = seeds[transformation_index];

        tensorflow::Tensor tensor(tensorflow::DataType::DT_INT64, tensorflow::TensorShape({seed.size()}));
        for (int i = 0; i < seed.size(); i++) {
            tensor.flat<long long>()(i) = seed[i];
        }
        feed_dict.push_back(std::make_pair(input_name, tensor));
    }

    feed_dict.push_back(std::make_pair(training_name, training_tensor));

    // create fetch outputs
    std::vector<std::string> fetch_outputs;
    std::vector<int> transformation_index_to_output;
    {
        int shift = 0;
        for (int transformation_index = 0; transformation_index < transformations.size(); transformation_index++) {
            transformation_index_to_output.push_back(shift);
            if (transformations_input_count[transformation_index] > 0) {
                fetch_outputs.push_back(transformations[transformation_index].output);
                shift++;
            }
        }
    }

    // run
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session->Run(feed_dict, fetch_outputs, {}, &outputs));

    // set outputs
    for(int node_index = batches[batches.size() - 1].nodes_end - 1; node_index >= batches[batches.size() - 2].nodes_end; node_index--) {
        auto &node = nodes[node_index];

        if (node.transformation != -1) {
            auto &transformation = transformations[node.transformation];
            auto &output = outputs[transformation_index_to_output[node.transformation]];
            for (int i = 0; i < transformation.output_size; i++) {
                float f = output.matrix<float>()(transformations_input_count[node.transformation] - 1, i);
                node.output.push_back(*((int*)&f));
            }
            transformations_input_count[node.transformation]--;
        }
    }
}

Eigen::VectorXf ComputationGraph::value(int index) {
    auto &node = nodes[index];
    Eigen::VectorXf result(node.output.size());
    for (int i = 0; i < node.output.size(); i++) {
        result(i) = node.output[i];
    }
    return result;
}

}
}