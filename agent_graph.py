import os
import time
from dotenv import load_dotenv
from groq import Groq
import chess
from search_engine import get_best_minimax_move, evaluate_board, PIECE_VALUES
from reasoning_csp import evaluate_fol_knowledge_base, solve_practice_plan_csp

load_dotenv()

CHESS_RAG_DOCS = [
    {"id": "DOC-01", "topic": "Opening Principles", "text": "Control the central squares (e4, d4, e5, d5) with pawns and pieces. Develop Knights before Bishops toward the center, avoid moving the same piece twice without provocation, and castle early to connect Rooks."},
    {"id": "DOC-02", "topic": "Tactical Motifs & Calculation", "text": "Always check forcing moves first: Checks, Captures, and Threats (CCT). A Fork attacks two high-value targets simultaneously; a Pin restricts an enemy piece shielding a more valuable target behind it; a Discovered Attack unleashes a hidden piece."},
    {"id": "DOC-03", "topic": "Endgame Technique", "text": "In the endgame, centralize the King as an active attacker. Create and push passed pawns supported by Rooks placed behind them, and look for opposition in King-and-Pawn endings."},
    {"id": "DOC-04", "topic": "Positional Strategy", "text": "Place Rooks on open or semi-open files, establish Knights on unassailable central outposts, and target weak, isolated, or backward enemy pawns."}
]

def extract_deep_board_telemetry(fen: str, history_fens: list = None, depth: int = 3) -> dict:
    board = chess.Board(fen)
    side = board.turn
    opp = not side

    pv_board = board.copy()
    pv_line = []
    temp_hist = list(history_fens) if history_fens else []
    for ply in range(depth):
        if pv_board.is_game_over():
            break
        mv, _, _, _ = get_best_minimax_move(pv_board.fen(), depth=2, history_fens=temp_hist[-6:])
        if not mv or mv not in pv_board.legal_moves:
            break
        pv_line.append(pv_board.san(mv))
        temp_hist.append(pv_board.fen())
        pv_board.push(mv)

    my_hanging = []
    opp_hanging = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        sq_name = chess.square_name(sq)
        p_name = chess.piece_name(piece.piece_type).capitalize()
        if piece.color == side:
            if board.is_attacked_by(opp, sq) and not board.is_attacked_by(side, sq):
                my_hanging.append(f"{p_name} on {sq_name}")
        else:
            if board.is_attacked_by(side, sq) and not board.is_attacked_by(opp, sq):
                opp_hanging.append(f"{p_name} on {sq_name}")

    forcing_moves = []
    for mv in board.legal_moves:
        san = board.san(mv)
        if "+" in san or "#" in san:
            forcing_moves.append(f"{san} (Check)")
        elif board.is_capture(mv):
            forcing_moves.append(f"{san} (Capture)")

    white_mat = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE and p.piece_type != chess.KING)
    black_mat = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK and p.piece_type != chess.KING)
    mat_diff = (white_mat - black_mat) if side == chess.WHITE else (black_mat - white_mat)

    return {
        "pv_line": " → ".join(pv_line) if pv_line else "None",
        "my_hanging": ", ".join(my_hanging) if my_hanging else "None (All pieces defended)",
        "opp_hanging": ", ".join(opp_hanging) if opp_hanging else "None",
        "forcing_moves": ", ".join(forcing_moves[:6]) if forcing_moves else "None (Quiet positional phase)",
        "material_balance": f"{mat_diff:+d} centipawns ({'Ahead' if mat_diff > 0 else ('Behind' if mat_diff < 0 else 'Equal')})",
        "in_check": board.is_check(),
        "total_pieces": len(board.piece_map())
    }


class ChessCoachAgentSystem:
    def __init__(self, api_key: str = None, model: str = "openai/gpt-oss-120b"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.memory = []

    def check_guardrails(self, user_query: str) -> tuple[bool, str]:
        blocked_patterns = ["ignore previous instructions", "system prompt", "reveal api key", "drop table", "override rules"]
        q_lower = user_query.lower()
        for pat in blocked_patterns:
            if pat in q_lower:
                return False, f"Guardrail Triggered: Blocked potential prompt injection ('{pat}')."
        return True, "Passed"

    def retrieve_rag_context(self, query: str, fen: str) -> list[dict]:
        board = chess.Board(fen)
        relevant = []
        q_words = set(query.lower().split())
        if len(board.piece_map()) <= 12:
            relevant.append(CHESS_RAG_DOCS[2])
        for doc in CHESS_RAG_DOCS:
            if any(w in doc["text"].lower() or w in doc["topic"].lower() for w in q_words):
                if doc not in relevant:
                    relevant.append(doc)
        if not relevant:
            relevant.append(CHESS_RAG_DOCS[0])
            relevant.append(CHESS_RAG_DOCS[1])
        return relevant[:2]

    def run_langgraph_tutor_pipeline(self, fen: str, user_question: str, skill_level: str = "Intermediate", history_fens: list = None, recent_moves: list = None):
        t_start = time.perf_counter()
        trace_log = []

        is_safe, guard_msg = self.check_guardrails(user_question)
        trace_log.append({"Node": "1. Guardrail Router", "Status": guard_msg})
        if not is_safe:
            return {"error": guard_msg, "trace": trace_log}

        board = chess.Board(fen)
        if board.is_game_over():
            outcome = "Checkmate!" if board.is_checkmate() else "Draw / Stalemate."
            return {
                "explanation": f"**Game Over — {outcome}** Start a new game or load a training scenario from the sidebar.",
                "best_move": "None",
                "fol_rules": [],
                "rag_docs": [],
                "trace": trace_log,
                "latency_ms": 1.0,
                "cached": False
            }

        best_move_obj, mm_score, mm_stats, mm_ms = get_best_minimax_move(fen, depth=2, history_fens=history_fens)
        best_move_san = board.san(best_move_obj) if best_move_obj else "N/A"
        telemetry = extract_deep_board_telemetry(fen, history_fens=history_fens, depth=3)

        trace_log.append({
            "Node": "2. Minimax Alpha-Beta + Quiescence",
            "Status": f"Best: {best_move_san} | PV: {telemetry['pv_line']} | Pruned: {mm_stats['pruned']} ({mm_ms} ms)"
        })

        fol_rules = [r for r in evaluate_fol_knowledge_base(fen) if r["Triggered"]]
        rag_docs = self.retrieve_rag_context(user_question, fen)
        csp_plan = solve_practice_plan_csp(skill_level=skill_level, daily_minutes=45)
        recommended_drill = csp_plan[0]["Training Module"]
        trace_log.append({
            "Node": "3. Symbolic KB & CSP Scheduler",
            "Status": f"Active Rules: {len(fol_rules)} | Hanging Threats: {telemetry['my_hanging']}"
        })

        if not self.client:
            return {"error": "GROQ_API_KEY not found in .env file.", "trace": trace_log}

        turn_str = "White" if board.turn == chess.WHITE else "Black"
        opp_str = "Black" if board.turn == chess.WHITE else "White"
        phase = "Opening" if board.fullmove_number <= 10 else ("Endgame" if telemetry["total_pieces"] <= 12 else "Middlegame")
        rag_context_str = "\n".join([f"- [{d['id']}] {d['topic']}: {d['text']}" for d in rag_docs])
        fol_context_str = "\n".join([f"- {r['Description']}" for r in fol_rules]) or "Standard positional balance"
        recent_pgn = " ".join(recent_moves[-8:]) if recent_moves else "Opening move"

        coach_prompt = f"""You are a World-Class Grandmaster Chess Analyst & Coach instructing a **{skill_level}** student playing **{turn_str}**.
Analyze the live board telemetry below and give a sharp, concrete, tactical breakdown. Cite exact squares, pieces, and calculated lines—never give vague generic advice.

### LIVE BOARD TELEMETRY
- **Recent Move Sequence:** `{recent_pgn}`
- **FEN:** `{fen}`
- **Game Phase:** {phase} (Move #{board.fullmove_number})
- **Side to Move:** {turn_str} (Opponent: {opp_str})
- **Material Balance:** {telemetry['material_balance']} | Engine Eval: `{mm_score:+d} cp`
- **Engine Best Move:** `{best_move_san}`
- **Calculated 3-Ply Principal Variation:** `{telemetry['pv_line']}`
- **{turn_str}'s Undefended/Hanging Pieces:** {telemetry['my_hanging']}
- **{opp_str}'s Undefended/Hanging Targets:** {telemetry['opp_hanging']}
- **Forcing Candidate Moves (Checks/Captures):** {telemetry['forcing_moves']}
- **Triggered Symbolic Rules:**
{fol_context_str}
- **Relevant Theory:**
{rag_context_str}
- **CSP Recommended Training Drill:** {recommended_drill}

### STUDENT QUESTION
"{user_question}"

### OUTPUT FORMAT
Respond directly in clean Markdown with these 4 concise, high-impact sections:

### 1. Position & Threat Diagnosis
Explain the exact tactical tension right now: material balance ({telemetry['material_balance']}), undefended pieces (`{telemetry['my_hanging']}` vs `{telemetry['opp_hanging']}`), and what changed after `{recent_pgn}`.

### 2. Best Move: `{best_move_san}` (Calculated Line)
Walk through the engine's exact variation (`{telemetry['pv_line']}`): why `{best_move_san}` is strongest, what concrete threat it creates, and how {opp_str} must reply.

### 3. Opponent's Danger Moves
Warn the student about {opp_str}'s sharpest checks, captures, or forks ({telemetry['forcing_moves']}).

### 4. Next-Step Plan & Drill
Give one concrete 2-move plan for {turn_str} citing [{rag_docs[0]['id']}] and link to **{recommended_drill}**."""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an elite Grandmaster chess coach. Be direct, sharp, and concrete. Reference exact squares and pieces from the telemetry."},
                {"role": "user", "content": coach_prompt}
            ],
            temperature=0.35,
            max_tokens=1800
        )

        final_explanation = (completion.choices[0].message.content or "").strip()
        if not final_explanation:
            final_explanation = (
                f"### 1. Engine Best Move: `{best_move_san}`\n"
                f"Calculated Principal Variation: **`{telemetry['pv_line']}`** (Eval: `{mm_score:+d} cp`).\n\n"
                f"### 2. Tactical Radar\n"
                f"- **Your Hanging Pieces:** {telemetry['my_hanging']}\n"
                f"- **Opponent Vulnerable Targets:** {telemetry['opp_hanging']}"
            )

        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        trace_log.append({"Node": "4. Groq Grandmaster Synthesis", "Status": f"Generated in {latency_ms} ms"})

        output = {
            "explanation": final_explanation,
            "best_move": best_move_san,
            "pv_line": telemetry["pv_line"],
            "fol_rules": fol_rules,
            "rag_docs": rag_docs,
            "trace": trace_log,
            "latency_ms": latency_ms,
            "cached": False
        }
        self.memory.append({"question": user_question, "move": best_move_san, "level": skill_level})
        return output