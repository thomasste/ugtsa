#include "games/omringa/game_state.h"
#include <iostream>

namespace games {
namespace omringa {

GameState::GameState(int board_size, int group_penalty, int min_bet, int max_bet)
    : games::game::GameState(0, 2), board_size(board_size), group_penalty(group_penalty), min_bet(min_bet), max_bet(max_bet),
      state(State::BET), bets({-1, -1}), chosen_player(-1), board(Eigen::MatrixXi::Zero(board_size, board_size)) {
//    board << 1, 2, 1, 1, 2, 2, 2,
//             2, 2, 1, 1, 1, 1, 2,
//             2, 2, 1, 2, 1, 1, 2,
//             2, 1, 1, 2, 1, 1, 2,
//             2, 2, 2, 1, 2, 2, 2,
//             1, 2, 1, 2, 1, 1, 1,
//             2, 1, 1, 1, 1, 1, 2;

    for (int x = 0; x < board_size; x++) {
        for (int y = 0; y < board_size; y++) {
            empty_positions.push_back({x, y});
        }
    }
}

GameState::GameState(const GameState &game_state)
    : GameState(game_state.board_size, game_state.group_penalty, game_state.min_bet, game_state.max_bet) {
    player = game_state.player;

    state = game_state.state;
    bets[0] = game_state.bets[0];
    bets[1] = game_state.bets[1];
    chosen_player = game_state.chosen_player;
    board = Eigen::MatrixXi(game_state.board);
    empty_positions = std::vector<Position>(game_state.empty_positions);
    move_history = std::vector<Move>(game_state.move_history);
}

int GameState::move_count() const {
    if (state == State::BET) {
        return max_bet - min_bet;
    } else if (state == State::NATURE) {
        return 2;
    } else {
        return empty_positions.size();
    }
}

void GameState::apply_move(int index) {
    Move move = get_move(index);
    move_history.push_back(move);

    if (state == State::BET) {
        bets[move.player] = move.value;
        if (bets[move.player ^ 1] == -1) {
            state = State::BET;
            player ^= 1;
        } else if (bets[0] == bets[1]) {
            state = State::NATURE;
            player = -1;
        } else {
            state = State::PLACE;
            player = bets[0] < bets[1];
        }
    } else if (state == State::NATURE) {
        state = State::PLACE;
        player = move.value;
        chosen_player = move.value;
    } else {
        std::swap(empty_positions[move.index], empty_positions.back());
        empty_positions.pop_back();
        board(move.position.y, move.position.x) = move.player + 1;
        player = move.player ^ 1;
    }
}

void GameState::undo_move() {
    Move move = move_history.back();
    move_history.pop_back();

    state = move.state;
    player = move.player;
    if (move.state == State::BET) {
        bets[move.player] = -1;
    } else if (move.state == State::NATURE) {
        chosen_player = -1;
    } else {
        empty_positions.push_back(move.position);
        std::swap(empty_positions[move.index], empty_positions.back());
        board(move.position.y, move.position.x) = 0;
    }
}

bool GameState::is_final() const {
    return empty_positions.empty();
}

Eigen::VectorXf GameState::payoff() const {
    Eigen::VectorXf result(2);
    result << (float) count_groups(0) * group_penalty, (float) count_groups(1) * group_penalty;

    for (int y = 0; y < board_size; y++) {
        for (int x = 0; x < board_size; x++) {
            if (board(y, x) == 1) {
                result[0] += 1;
            } else if (board(y, x) == 2) {
                result[1] += 1;
            }
        }
    }

    if (bets[0] < bets[1]) {
        result[0] += (float) bets[0] + 0.5;
    } else if (bets[0] > bets[1]) {
        result[1] += (float) bets[1] + 0.5;
    } else {
        result[chosen_player ^ 1] += bets[chosen_player ^ 1] + 0.5;
    }

    return result;
}

Eigen::MatrixXf GameState::as_matrix() const {
    return board.cast<float>();
}

GameState *GameState::GameState::copy() const {
    return new GameState(*this);
}

void GameState::serialize(std::ostream &stream) const {
    stream << "player " << player << std::endl
           << "chosen_player " << chosen_player << std::endl
           << "state " << state << std::endl
           << "bets " << bets[0] << " " << bets[1] << std::endl
           << board << std::endl
           << "moves" << std::endl;
    for (int i = 0; i < move_count(); i++) {
        stream << i << " " << get_move(i) << ", ";
    }
}

GameState::Move GameState::get_move(int index) const {
    Move move;
    move.state = state;
    move.player = player;

    if (state == State::BET) {
        move.value = min_bet + index;
        move.index = -1;
    } else if (state == State::NATURE) {
        move.value = index;
        move.index = -1;
    } else {
        move.position = empty_positions[index];
        move.index = index;
    }

    return move;
}

int GameState::count_groups(int player) const {
    player = player + 1;

    int result = 0;
    std::vector<Position> stack;
    Eigen::MatrixXd visited = Eigen::MatrixXd::Zero(board_size, board_size);

    for (int y = 0; y < board_size; y++) {
        for (int x = 0; x < board_size; x++) {
            if (visited(y, x) == 0 && board(y, x) == player) {
                visited(y, x) = 1;
                result += 1;

                stack.push_back({x, y});
                while (!stack.empty()) {
                    Position p = stack.back();
                    stack.pop_back();

                    int dy[] = {-1, 0, 0, 1};
                    int dx[] = {0, -1, 1, 0};

                    for (int i = 0; i < 4; i++) {
                        int ny = p.y + dy[i];
                        int nx = p.x + dx[i];

                        if (0 <= nx && nx < board_size && 0 <= ny && ny < board_size &&
                                visited(ny, nx) == 0 && board(ny, nx) == player) {
                            visited(ny, nx) = 1;
                            stack.push_back({nx, ny});
                        }
                    }
                }
            }
        }
    }

    return result;
}

std::ostream& operator<<(std::ostream& stream, const GameState::Position& position) {
    return stream << "(" << position.x << ", " << position.y << ")";
}

std::ostream& operator<<(std::ostream& stream, const GameState::State& state) {
    if (state == GameState::State::BET) stream << "bet";
    else if (state == GameState::State::NATURE) stream << "nature";
    else if (state == GameState::State::PLACE) stream << "place";

    return stream;
}

std::ostream& operator<<(std::ostream& stream, const GameState::Move& move) {
    stream << "player " << move.player << " " << move.state << " ";

    if (move.state == GameState::State::BET) {
        stream << move.value;
    } else if (move.state == GameState::State::NATURE) {
        stream << move.value;
    } else {
        stream << move.position;
    }

    return stream;
}

std::ostream& operator<<(std::ostream& stream, const GameState& game_state) {
    game_state.serialize(stream);
    return stream;
}

}
}
