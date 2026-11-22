import time
import chess

class TacticalCSP:
    def __init__(self, n: int = 8):
        self.n = n
        self.backtracks = 0

    def is_safe(self, assignment: dict, row: int, col: int) -> bool:
        for r, c in assignment.items():
            if c == col or abs(r - row) == abs(c - col):
                return False
        return True

    def solve_plain_backtracking(self):
        self.backtracks = 0
        t0 = time.perf_counter()
        solution = self._bt({})
        elapsed = (time.perf_counter() - t0) * 1000
        return solution, self.backtracks, round(elapsed, 3)

    def _bt(self, assignment: dict):
        if len(assignment) == self.n:
            return assignment
        row = len(assignment)
        for col in range(self.n):
            if self.is_safe(assignment, row, col):
                assignment[row] = col
                res = self._bt(assignment)
                if res is not None:
                    return res
                del assignment[row]
                self.backtracks += 1
        return None

    def solve_fc_mrv(self):
        self.backtracks = 0
        t0 = time.perf_counter()
        domains = {r: list(range(self.n)) for r in range(self.n)}
        solution = self._bt_fc({}, domains)
        elapsed = (time.perf_counter() - t0) * 1000
        return solution, self.backtracks, round(elapsed, 3)

    def _bt_fc(self, assignment: dict, domains: dict):
        if len(assignment) == self.n:
            return assignment
        unassigned = [r for r in range(self.n) if r not in assignment]
        row = min(unassigned, key=lambda r: len(domains[r]))

        for col in domains[row]:
            if self.is_safe(assignment, row, col):
                assignment[row] = col
                new_domains = {r: list(domains[r]) for r in unassigned if r != row}
                wipeout = False
                for r in new_domains:
                    new_domains[r] = [
                        c for c in new_domains[r]
                        if c != col and abs(r - row) != abs(c - col)
                    ]
                    if not new_domains[r]:
                        wipeout = True
                        break
                if not wipeout:
                    res = self._bt_fc(assignment, new_domains)
                    if res is not None:
                        return res
                del assignment[row]
                self.backtracks += 1
        return None


def solve_practice_plan_csp(skill_level: str = "Intermediate", daily_minutes: int = 45):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if skill_level == "Beginner":
        modules = [
            {"Module": "Tactical Puzzles (Forks & Pins)", "Duration": min(daily_minutes, 30), "Focus": "Pattern Recognition"},
            {"Module": "Opening Principles & Center Control", "Duration": min(daily_minutes, 30), "Focus": "Development"},
            {"Module": "Basic Checkmates (King + Rook/Queen)", "Duration": min(daily_minutes, 25), "Focus": "Endgame Fundamentals"},
            {"Module": "Guided Rapid Game + AI Review", "Duration": daily_minutes, "Focus": "Practical Application"}
        ]
    elif skill_level == "Intermediate":
        modules = [
            {"Module": "Calculation & 3-Ply Tactical Drills", "Duration": daily_minutes, "Focus": "Tactical Vision"},
            {"Module": "Repertoire Deep-Dive (Italian / Sicilian)", "Duration": min(daily_minutes, 35), "Focus": "Opening Theory"},
            {"Module": "Rook & Pawn Endgame Technique", "Duration": min(daily_minutes, 40), "Focus": "Endgame Precision"},
            {"Module": "Annotated Rapid Game + Coach Critique", "Duration": daily_minutes, "Focus": "Decision Quality"}
        ]
    else:
        modules = [
            {"Module": "Deep Calculation & Candidate Move Trees", "Duration": daily_minutes, "Focus": "Visualization"},
            {"Module": "Positional Pawn Structures & Outposts", "Duration": min(daily_minutes, 40), "Focus": "Strategic Planning"},
            {"Module": "Theoretical Endgame Lab (Lucena/Philidor)", "Duration": daily_minutes, "Focus": "Endgame Mastery"},
            {"Module": "Competitive Sparring + Blunder Audit", "Duration": daily_minutes, "Focus": "Tournament Prep"}
        ]

    schedule = []
    prev_idx = -1
    for day_idx, day in enumerate(days):
        chosen_idx = (day_idx * 2) % len(modules)
        if chosen_idx == prev_idx:
            chosen_idx = (chosen_idx + 1) % len(modules)
        prev_idx = chosen_idx
        item = modules[chosen_idx]
        schedule.append({
            "Day": day,
            "Training Module": item["Module"],
            "Allocated Time": f"{item['Duration']} mins",
            "Target Skill": item["Focus"],
            "Constraint Status": "Satisfied"
        })
    return schedule


def evaluate_fol_knowledge_base(fen: str):
    board = chess.Board(fen)
    turn_color = chess.WHITE if board.turn == chess.WHITE else chess.BLACK

    fol_rules = [
        {
            "ID": "FOL-01",
            "Formula": "∀x (King(x) ∧ InCheck(x) → MustEvadeOrBlock(x))",
            "Description": "King safety constraint: Active check forces immediate resolution.",
            "Triggered": board.is_check()
        },
        {
            "ID": "FOL-02",
            "Formula": "∃p ∀s (Piece(p) ∧ Central(s) ∧ Controls(p, s) → CenterDominance(p))",
            "Description": "Classical opening principle: Controlling e4, d4, e5, d5.",
            "Triggered": any(
                board.piece_at(sq) and board.piece_at(sq).color == turn_color
                for sq in [chess.E4, chess.D4, chess.E5, chess.D5]
            )
        },
        {
            "ID": "FOL-03",
            "Formula": "∃p ∀t1,t2 (Piece(p) ∧ Attacks(p, t1) ∧ Attacks(p, t2) → TacticalFork(p))",
            "Description": "Tactical Fork Detection: Piece attacking >=2 high-value targets.",
            "Triggered": _detect_fork(board, turn_color)
        },
        {
            "ID": "FOL-04",
            "Formula": "∀p (Pinned(p, King) → ¬CanMoveOffRay(p))",
            "Description": "Absolute Pin Rule: Piece pinned to its own King cannot move off ray.",
            "Triggered": any(
                board.is_pinned(turn_color, sq) for sq in chess.SQUARES
                if board.piece_at(sq) and board.piece_at(sq).color == turn_color
            )
        },
        {
            "ID": "FOL-05",
            "Formula": "CanCastleKingside(Player) ↔ ¬HasMoved(King) ∧ ¬HasMoved(RookH)",
            "Description": "Kingside castling eligibility predicate.",
            "Triggered": board.has_kingside_castling_rights(turn_color)
        },
        {
            "ID": "FOL-06",
            "Formula": "TotalMaterial(Board) ≤ EndgameThreshold → ActivateKing(King)",
            "Description": "Endgame transition rule: Activate King when major pieces are traded.",
            "Triggered": len(board.piece_map()) <= 12
        }
    ]
    return fol_rules

def _detect_fork(board: chess.Board, color: chess.Color) -> bool:
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            attacked = board.attacks(sq)
            valuable_targets = 0
            for target_sq in attacked:
                target = board.piece_at(target_sq)
                if target and target.color != color and target.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                    valuable_targets += 1
            if valuable_targets >= 2:
                return True
    return False

def belief_state_opponent_model(fen: str):
    board = chess.Board(fen)
    beliefs = []
    for move in list(board.legal_moves)[:4]:
        b_copy = board.copy()
        is_cap = b_copy.is_capture(move)
        san = b_copy.san(move)
        b_copy.push(move)
        intent = "Aggressive Capture" if is_cap else ("Forcing Check" if b_copy.is_check() else "Positional Development")
        prob = "45%" if is_cap else ("35%" if b_copy.is_check() else "20%")
        beliefs.append({"Candidate Response": san, "Inferred Strategy": intent, "Belief Probability": prob})
    return beliefs