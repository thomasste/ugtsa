#include "ugtsa/algorithms/generalized_mcts/algorithm.h"

#include <algorithm>
#include <iostream>

namespace algorithms {
namespace generalized_mcts {

Algorithm::Algorithm(games::game::GameState *game_state, int worker_count, int grow_factor, std::vector<int> removed_root_moves)
    : algorithms::algorithm::Algorithm(game_state), grow_factor(grow_factor), removed_root_moves(removed_root_moves) {
    for (int i = 0; i < worker_count; i++) {
        workers.push_back({0, Direction::DOWN, std::unique_ptr<games::game::GameState>(game_state->copy()), -1});
    }
}

void Algorithm::improve() {
    if (tree.empty()) {
        tree.push_back({0, -1, {-1, -1}, empty_statistic(game_state), -1});
        run_batch();
    }

    std::vector<int> empty_statistics;
    std::vector<int> move_rates;
    std::vector<int> game_state_as_updates;
    std::vector<int> updated_statistics;
    std::vector<int> updated_updates;

    std::vector<int> worker_indices; for (int i = 0; i < workers.size(); i++) worker_indices.push_back(i);
    std::vector<int> up_workers;
    std::vector<int> down_workers;
    auto worker_node_comparator = [this](const int a, const int b) { return this->workers[a].node < this->workers[b].node; };
    std::copy_if(worker_indices.begin(), worker_indices.end(), std::back_inserter(up_workers), [this](int a) { return this->workers[a].direction == Direction::UP; });
    std::copy_if(worker_indices.begin(), worker_indices.end(), std::back_inserter(down_workers), [this](int a) { return this->workers[a].direction == Direction::DOWN; });
    std::sort(up_workers.begin(), up_workers.end(), worker_node_comparator);
    std::sort(down_workers.begin(), down_workers.end(), worker_node_comparator);

    // down
    for (auto it = down_workers.begin(); it != down_workers.end();) {
        std::vector<int> group; group.push_back(*it); it++;
        while (it != down_workers.end() && workers[group.back()].node == workers[*it].node) {
            group.push_back(*it);
            it++;
        }

        //std::cout << "group size" << group.size() << workers[0].node << std::endl;

        Node &node = tree[workers[group[0]].node];
        games::game::GameState *game_state = workers[group[0]].game_state.get();

        if (node.children[0] > 0) {
            for (int i = node.children[0]; i < node.children[1]; i++) {
                Node &child = tree[i];
                if (child.move_rate_cache == -1) {
                    move_rates.push_back(move_rate(node.statistic, child.statistic));
                }
            }
        } else {
            if (node.number_of_visits >= grow_factor && !game_state->is_final()) {
                int move_count = game_state->move_count();
                for (int i = 0; i < move_count; i++) {
                    game_state->apply_move(i);
                    empty_statistics.push_back(empty_statistic(game_state));
                    game_state->undo_move();
                }
            } else {
                for (int i = 0; i < group.size(); i++) {
                    game_state_as_updates.push_back(game_state_as_update(game_state));
                }
            }
        }
    }

    // up
    for (auto it = up_workers.begin(); it != up_workers.end();) {
        std::vector<int> group; group.push_back(*it); it++;
        while (it != up_workers.end() && workers[group.back()].node == workers[*it].node) {
            group.push_back(*it);
            it++;
        }

        Node &node = tree[workers[group[0]].node];
        games::game::GameState *game_state = workers[group[0]].game_state.get();

        std::vector<int> updates;
        for (auto it = group.begin(); it != group.end(); it++) {
            updates.push_back(workers[*it].update);
        }
        updated_statistics.push_back(updated_statistic(node.statistic, updates));

        for (auto it = group.begin(); it != group.end(); it++) {
            updated_updates.push_back(updated_update(workers[*it].update, node.statistic));
        }
    }

    run_batch();

    std::reverse(empty_statistics.begin(), empty_statistics.end());
    std::reverse(move_rates.begin(), move_rates.end());
    std::reverse(game_state_as_updates.begin(), game_state_as_updates.end());
    std::reverse(updated_statistics.begin(), updated_statistics.end());
    std::reverse(updated_updates.begin(), updated_updates.end());

    // down
    for (auto it = down_workers.begin(); it != down_workers.end();) {
        std::vector<int> group; group.push_back(*it); it++;
        while (it != down_workers.end() && workers[group.back()].node == workers[*it].node) {
            group.push_back(*it);
            it++;
        }

        Node &node = tree[workers[group[0]].node];
        games::game::GameState *game_state = workers[group[0]].game_state.get();

        if (node.children[0] > 0) {
            for (int i = node.children[0]; i < node.children[1]; i++) {
                Node &child = tree[i];
                if (child.move_rate_cache == -1) {
                    child.move_rate_cache = move_rates.back();
                    move_rates.pop_back();
                }
            }
            down_move_case(group);
        } else {
            if (node.number_of_visits >= grow_factor && !game_state->is_final()) {
                std::vector<int> result;
                int move_count = game_state->move_count();
                for (int i = 0; i < move_count; i++) {
                    result.push_back(empty_statistics.back());
                    empty_statistics.pop_back();
                }
                down_expand_case(group, result);
            } else {
                std::vector<int> result;
                for (int i = 0; i < group.size(); i++) {
                    result.push_back(game_state_as_updates.back());
                    game_state_as_updates.pop_back();
                }
                down_leaf_case(group, result);
            }
        }
    }

    // up
    for (auto it = up_workers.begin(); it != up_workers.end();) {
        std::vector<int> group; group.push_back(*it); it++;
        while (it != up_workers.end() && workers[group.back()].node == workers[*it].node) {
            group.push_back(*it);
            it++;
        }

        int result1 = updated_statistics.back();
        updated_statistics.pop_back();

        std::vector<int> result2;
        for (auto it = group.begin(); it != group.end(); it++) {
            result2.push_back(updated_updates.back());
            updated_updates.pop_back();
        }
        up_case(group, result1, result2);
    }
}

std::vector<int> Algorithm::move_rates() {
    assert(tree.size() > 1);
    std::vector<int> result;

    Node& root = tree[0];
    for (int i = root.children[0]; i < root.children[1]; i++) {
        Node &child = tree[i];
        result.push_back(move_rate(root.statistic, child.statistic));
    }
    run_batch();

    return result;
}

void Algorithm::serialize(std::ostream &stream) const {
    stream << "WORKERS" << std::endl;
    for (auto& worker : workers) stream << worker << std::endl;
    stream << "NODES " << tree.size() << std::endl;
    for (int i = 0; i < 20 && i < tree.size(); i++) {
        stream << tree[i] << std::endl;
    }
}

void Algorithm::down_move_case(std::vector<int> &group) {
    Node& node = tree[workers[group[0]].node];
    games::game::GameState *game_state = workers[group[0]].game_state.get();

    std::vector<float> probabilities;

    //std::cout << "PROBABILITIES" << std::endl;

    if (game_state->player != -1) {
        for (int i = node.children[0]; i < node.children[1]; i++) {
            Node& child = tree[i];
            probabilities.push_back(value(child.move_rate_cache)(game_state->player));
        }

        if (node.parent == -1) {
            for (auto it = removed_root_moves.begin(); it != removed_root_moves.end(); it++) {
                probabilities[*it] = 0.;
            }
        }

        // for (auto p : probabilities) std::cout << p << " " << std::endl;

        float sum = 0.;
        for (auto &probability : probabilities) sum += probability;
        for (auto &probability : probabilities) probability /= sum;
        // for (auto p : probabilities) std::cout << p << " " << std::endl;
    } else {
        int children_count = node.children[1] - node.children[0];
        for (int i = 0; i < children_count; i++)
            probabilities.push_back(1. / children_count);
    }

    std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

    for (auto it = group.begin(); it != group.end(); it++) {
        Worker &worker = workers[*it];
        int move = distribution(generator);
        worker.node = node.children[0] + move;
        worker.game_state->apply_move(move);
    }
}

void Algorithm::down_expand_case(std::vector<int> &group, std::vector<int> &empty_statistics) {
    Node& node = tree[workers[group[0]].node];

    node.children[0] = tree.size();
    node.children[1] = tree.size() + empty_statistics.size();

    for (auto empty_statistic : empty_statistics)
        tree.push_back({0, workers[group[0]].node, {-1, -1}, empty_statistic, -1});
}

void Algorithm::down_leaf_case(std::vector<int> &group, std::vector<int> &game_state_as_updates) {
    for (int i = 0; i < group.size(); i++) {
        Worker &worker = workers[group[i]];
        worker.direction = Direction::UP;
        worker.update = game_state_as_updates[i];
    }
}

void Algorithm::up_case(std::vector<int> &group, int updated_statistic, std::vector<int> &updated_updates) {
    Node& node = tree[workers[group[0]].node];

    node.move_rate_cache = -1;
    node.statistic = updated_statistic;
    node.number_of_visits += group.size();

    for (int i = 0; i < group.size(); i++) {
        Worker &worker = workers[group[i]];
        worker.update = updated_updates[i];
    }

    if (node.parent != -1) {
        for (auto i : group) {
            Worker &worker = workers[i];
            worker.game_state->undo_move();
            worker.node = node.parent;
        }
    } else {
        for (auto i : group) {
            Worker &worker = workers[i];
            worker.update = -1;
            worker.direction = Direction::DOWN;
        }
    }
}

std::ostream& operator<<(std::ostream& stream, const Algorithm::Direction& direction) {
    if (direction == Algorithm::Direction::UP) stream << "up";
    else if (direction == Algorithm::Direction::DOWN) stream << "down";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const Algorithm::Worker& worker) {
    return stream << worker.direction << " node " << worker.node << " update " << worker.update << std::endl
                  << *(worker.game_state);
}

std::ostream& operator<<(std::ostream& stream, const Algorithm::Node& node) {
    return stream << "num_of_vis " << node.number_of_visits << " parent " << node.parent
                  << " children (" << node.children[0] << ", " << node.children[1] << ")"
                  << " statistic " << node.statistic << " move_rate_cache " << node.move_rate_cache;
}

std::ostream& operator<<(std::ostream& stream, const Algorithm& algorithm) {
    algorithm.serialize(stream);
    return stream;
}

}
}
