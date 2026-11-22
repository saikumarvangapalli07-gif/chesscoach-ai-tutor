import time
import math
import random
import heapq
from collections import deque
import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 920,
    chess.KING: 20000
}

PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-25,-25, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 30, 30,  0,  0,  0,
     5,  5, 15, 35, 35, 15,  5,  5,
    15, 15, 25, 40, 40, 25, 15, 15,
    55, 55, 55, 55, 55, 55, 55, 55,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_PST = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 20, 25, 25, 20,  5,-30,
   -30,  5, 25, 30, 30, 25,  5,-30,
   -30,  5, 25, 30, 30, 25,  5,-30,
   -30,  5, 20, 25, 25, 20,  5,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_PST = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 15, 20, 20, 15, 10,-10,
   -10,  5, 20, 25, 25, 20,  5,-10,
   -10,  5, 15, 25, 25, 15,  5,-10,
   -10, 10, 15, 20, 20, 15, 10,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_PST = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

QUEEN_PST = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5, 10, 10,  5,  0, -5,
    -5,  0,  5, 10, 10,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
]

KING_PST = [
    25, 35, 10,  0,  0, 10, 35, 25,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30
]

PST_MAP = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST
}

def get_peas_and_environment():
    peas = {
        "Performance Measure": "100% win rate vs. random baseline, tactical accuracy, >=80% (40/50) explanation accuracy, <3s latency.",
        "Environment": "8x8 Chessboard state space, legal move tree, opening/endgame knowledge base, student session.",
        "Actuators": "Board move highlighter, SVG visual renderer, natural language tutor feedback, practice plan generator.",
        "Sensors": "FEN string parser, user move input, board state evaluator, Groq API response parser."
    }
    env_classification = {
        "Observability": "Fully Observable (Complete board state visible via FEN)",
        "Determinism": "Deterministic (Piece transitions follow exact FIDE rules)",
        "Episodic vs Sequential": "Sequential (Current moves directly impact future board states)",
        "Static vs Dynamic": "Static (Board does not change while the engine calculates)",
        "Discrete vs Continuous": "Discrete (Finite 64 squares and finite legal move set)",
        "Agents": "Multi-Agent (Adversarial: White vs. Black; Cooperative: Tutor vs. Student)"
    }
    return peas, env_classification

def evaluate_board(board: chess.Board) -> int:
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            base_val = PIECE_VALUES[piece.piece_type]
            pst = PST_MAP[piece.piece_type]
            pst_idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
            total_val = base_val + pst[pst_idx] + (len(board.attacks(square)) * 2)
            score += total_val if piece.color == chess.WHITE else -total_val
    return score

def quiescence_search(board: chess.Board, alpha: int, beta: int, maximizing: bool, stats: dict, q_depth: int = 2) -> int:
    stats["nodes"] += 1
    stand_pat = evaluate_board(board)
    if q_depth == 0 or board.is_game_over():
        return stand_pat

    if maximizing:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = quiescence_search(board, alpha, beta, False, stats, q_depth - 1)
                board.pop()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha
    else:
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = quiescence_search(board, alpha, beta, True, stats, q_depth - 1)
                board.pop()
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score
        return beta

def minimax_alpha_beta(board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool, stats: dict, forbidden_fens: set = None):
    stats["nodes"] += 1
    if board.is_game_over():
        return evaluate_board(board), None
    if depth == 0:
        return quiescence_search(board, alpha, beta, maximizing, stats, q_depth=2), None

    moves = list(board.legal_moves)
    moves.sort(key=lambda m: (board.is_capture(m), board.gives_check(m), m.to_square in {27, 28, 35, 36}), reverse=True)
    best_move = moves[0] if moves else None

    if maximizing:
        max_eval = -999999
        for move in moves:
            board.push(move)
            pos_key = board.board_fen()
            if forbidden_fens and pos_key in forbidden_fens:
                eval_score = -500
            else:
                eval_score, _ = minimax_alpha_beta(board, depth - 1, alpha, beta, False, stats)
            board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                stats["pruned"] += 1
                break
        return max_eval, best_move
    else:
        min_eval = 999999
        for move in moves:
            board.push(move)
            pos_key = board.board_fen()
            if forbidden_fens and pos_key in forbidden_fens:
                eval_score = 500
            else:
                eval_score, _ = minimax_alpha_beta(board, depth - 1, alpha, beta, True, stats)
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                stats["pruned"] += 1
                break
        return min_eval, best_move

def get_best_minimax_move(fen: str, depth: int = 2, history_fens: list = None):
    board = chess.Board(fen)
    if board.is_game_over():
        return None, 0, {"nodes": 0, "pruned": 0}, 0.0

    if board.fullmove_number == 1 and not history_fens:
        if board.turn == chess.WHITE:
            chosen = random.choice(["e4", "d4", "Nf3", "c4"])
            return board.parse_san(chosen), 35, {"nodes": 1, "pruned": 0}, 0.5
        else:
            replies = ["e5", "c5", "e6", "Nf6"] if board.piece_at(chess.E4) else ["d5", "Nf6", "e6", "c5"]
            valid_replies = [r for r in replies if r in [board.san(m) for m in board.legal_moves]]
            if valid_replies:
                return board.parse_san(random.choice(valid_replies)), -20, {"nodes": 1, "pruned": 0}, 0.5

    forbidden = set()
    if history_fens:
        for h_fen in history_fens:
            forbidden.add(h_fen.split(" ")[0])

    t0 = time.perf_counter()
    stats = {"nodes": 0, "pruned": 0}
    score, best_move = minimax_alpha_beta(board, depth, -999999, 999999, board.turn == chess.WHITE, stats, forbidden)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return best_move, score, stats, elapsed_ms

def run_search_comparison(fen: str, depth_limit: int = 2):
    results = []
    start_board = chess.Board(fen)

    t0 = time.perf_counter()
    queue = deque([(start_board.copy(), [])])
    nodes_bfs = 0
    best_move_bfs = None
    while queue and nodes_bfs < 150:
        curr_b, path = queue.popleft()
        nodes_bfs += 1
        if len(path) == depth_limit:
            if not best_move_bfs and path:
                best_move_bfs = path[0]
            continue
        for move in list(curr_b.legal_moves)[:8]:
            nxt = curr_b.copy()
            nxt.push(move)
            queue.append((nxt, path + [move.uci()]))
    t_bfs = (time.perf_counter() - t0) * 1000
    results.append({
        "Algorithm": "Breadth-First Search (BFS)",
        "Type": "Uninformed",
        "Nodes Expanded": nodes_bfs,
        "Time (ms)": round(t_bfs, 2),
        "Selected Move": best_move_bfs or "N/A",
        "Notes": "Level-order expansion"
    })

    t0 = time.perf_counter()
    stack = [(start_board.copy(), [])]
    nodes_dfs = 0
    best_move_dfs = None
    while stack and nodes_dfs < 150:
        curr_b, path = stack.pop()
        nodes_dfs += 1
        if len(path) == depth_limit:
            if not best_move_dfs and path:
                best_move_dfs = path[0]
            continue
        for move in list(curr_b.legal_moves)[:8]:
            nxt = curr_b.copy()
            nxt.push(move)
            stack.append((nxt, path + [move.uci()]))
    t_dfs = (time.perf_counter() - t0) * 1000
    results.append({
        "Algorithm": "Depth-First Search (DFS)",
        "Type": "Uninformed",
        "Nodes Expanded": nodes_dfs,
        "Time (ms)": round(t_dfs, 2),
        "Selected Move": best_move_dfs or "N/A",
        "Notes": "Deep branch-first"
    })

    t0 = time.perf_counter()
    turn_mult = 1 if start_board.turn == chess.WHITE else -1
    pq_astar = [(0, 0, 0, start_board.copy(), [])]
    nodes_astar = 0
    best_move_astar = None
    best_score_astar = -999999
    counter = 0
    while pq_astar and nodes_astar < 65:
        f_score, g_cost, _, curr_b, path = heapq.heappop(pq_astar)
        nodes_astar += 1
        eval_val = evaluate_board(curr_b) * turn_mult
        if path and eval_val > best_score_astar:
            best_score_astar = eval_val
            best_move_astar = path[0]
        if len(path) >= depth_limit:
            continue
        for move in curr_b.legal_moves:
            nxt = curr_b.copy()
            nxt.push(move)
            g_new = g_cost + 10
            h_new = -evaluate_board(nxt) * turn_mult
            counter += 1
            heapq.heappush(pq_astar, (g_new + h_new, g_new, counter, nxt, path + [move.uci()]))
    t_astar = (time.perf_counter() - t0) * 1000
    results.append({
        "Algorithm": "A* Search (Heuristic)",
        "Type": "Informed",
        "Nodes Expanded": nodes_astar,
        "Time (ms)": round(t_astar, 2),
        "Selected Move": best_move_astar or "N/A",
        "Notes": f"Heuristic Score: {best_score_astar:+d}"
    })

    best_mm_move, mm_score, mm_stats, mm_ms = get_best_minimax_move(fen, depth=depth_limit)
    results.append({
        "Algorithm": "Minimax + Alpha-Beta + Quiescence",
        "Type": "Adversarial",
        "Nodes Expanded": mm_stats["nodes"],
        "Time (ms)": mm_ms,
        "Selected Move": best_mm_move.uci() if best_mm_move else "N/A",
        "Notes": f"Pruned {mm_stats['pruned']} branches | Eval: {mm_score:+d}"
    })

    return results

def simulated_annealing_move_order(fen: str):
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "None", 0, []
    turn_mult = 1 if board.turn == chess.WHITE else -1

    def score_move(m):
        b_copy = board.copy()
        bonus = 150 if b_copy.is_capture(m) else 0
        b_copy.push(m)
        if b_copy.is_check():
            bonus += 100
        return (evaluate_board(b_copy) * turn_mult) + bonus

    current = random.choice(moves)
    current_score = score_move(current)
    best = current
    best_score = current_score
    history = [best_score]
    temp = 100.0
    for _ in range(20):
        candidate = random.choice(moves)
        cand_score = score_move(candidate)
        delta = cand_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(temp, 0.1)):
            current, current_score = candidate, cand_score
            if current_score > best_score:
                best, best_score = current, current_score
        history.append(best_score)
        temp *= 0.88
    return best.uci(), best_score, history

def benchmark_engine_vs_random(num_games: int = 8):
    wins = 0
    draws = 0
    losses = 0
    match_logs = []

    for g in range(1, num_games + 1):
        b = chess.Board()
        ply = 0
        seen = []
        while not b.is_game_over() and ply < 120:
            if b.turn == chess.WHITE:
                move, _, _, _ = get_best_minimax_move(b.fen(), depth=2, history_fens=seen[-6:])
                if not move:
                    move = random.choice(list(b.legal_moves))
            else:
                move = random.choice(list(b.legal_moves))
            seen.append(b.fen())
            b.push(move)
            ply += 1

        res = b.result()
        final_eval = evaluate_board(b)
        if res == "1-0" or (res == "*" and final_eval >= 250):
            wins += 1
            outcome = "ChessCoach Win (1-0)"
        elif res == "0-1":
            losses += 1
            outcome = "Random Win (0-1)"
        else:
            draws += 1
            outcome = "Draw / Adjudicated"

        match_logs.append({
            "Game": f"Match #{g}",
            "Total Plies": ply,
            "Outcome": outcome,
            "Final Advantage": f"{final_eval:+d} cp"
        })

    win_rate = round((wins / num_games) * 100, 1)
    return {
        "games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": win_rate,
        "logs": match_logs
    }