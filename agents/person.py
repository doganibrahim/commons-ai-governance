from mesa import Agent
import random
from modules.psychology import PsychologyModel

# ─────────────────────────────────────────────────────────────────────────────
# agent types and their profiles
# ─────────────────────────────────────────────────────────────────────────────
AGENT_TYPE_PROFILES = {
    "ideal": {
        "base_trust":         65,
        "autonomy":           1.0,
        "satisfaction":       0.7,
        "max_usage_duration": 5,
        # w1=trust, w2=satisfaction, w3=autonomy, w4=scarcity
        "w1_range": (0.40, 0.50),
        "w2_range": (0.30, 0.45),
        "w3_range": (0.15, 0.25),
        "w4_range": (0.05, 0.15),
    },
    "standard": {
        "base_trust":         50,
        "autonomy":           0.7,
        "satisfaction":       0.5,
        "max_usage_duration": 5,
        "w1_range": (0.30, 0.50),
        "w2_range": (0.30, 0.50),
        "w3_range": (0.10, 0.30),
        "w4_range": (0.10, 0.30),
    },
    "toxic": {
        "base_trust":         30,
        "autonomy":           0.4,
        "satisfaction":       0.3,
        "max_usage_duration": 10,   # naturally, the resource is used for a longer time
        "w1_range": (0.15, 0.30),
        "w2_range": (0.20, 0.35),
        "w3_range": (0.10, 0.20),
        "w4_range": (0.25, 0.35),   # very sensitive to scarcity → competitive
    },
}

# how many steps in a row a resource is not available, the "frustration" is triggered
FRUSTRATION_THRESHOLD = 3


class PersonAgent(Agent):
    def __init__(self, model, agent_type: str = "standard"):
        super().__init__(model)

        self.agent_type = agent_type
        profile = AGENT_TYPE_PROFILES.get(agent_type, AGENT_TYPE_PROFILES["standard"])

        # ── resource usage status ────────────────────────────────────────────
        self.current_resource = None
        self.usage_duration = 0
        self.wait_time = 0
        self.is_defecting = False
        self.max_usage_duration = profile["max_usage_duration"]

        # ── cumulative usage (for Gini coefficient calculation) ──────────────
        self.cumulative_usage = 0

        # ── frustration counter ──────────────────────────────────────────────
        self.frustration_counter = 0

        # ── psychological variables ───────────────────────────────────────────
        # trust: AI penalty applied based on the model's system type
        is_ai = model.system_type in ("ai_advisory", "ai_autonomous", "integrated")
        self.trust = PsychologyModel.calculate_initial_trust(
            base_trust=profile["base_trust"],
            is_ai_system=is_ai,
        )
        self.autonomy = profile["autonomy"]
        self.satisfaction = profile["satisfaction"]

        # perceived community fairness from neighbor observation (0-1)
        self.perceived_community_fairness = 0.5

        # ── cooperation decision weights (randomly from type-specific ranges) ──
        self.weights = (
            random.uniform(*profile["w1_range"]),
            random.uniform(*profile["w2_range"]),
            random.uniform(*profile["w3_range"]),
            random.uniform(*profile["w4_range"]),
        )
        self.last_iforest_label = 1
        self.last_sanction_tick = -1

    # ─────────────────────────────────────────────────────────────────────────
    # main decision loop
    # ─────────────────────────────────────────────────────────────────────────

    def step(self):
        """
        Main decision loop:
        - if not holding a resource: wait, observe, decide, request.
        - if holding a resource: use, release when duration is over.
        """
        if self.current_resource is None:
            self.wait_time += 1
            self.request_resource()
        else:
            self.use_resource()

    # ─────────────────────────────────────────────────────────────────────────
    # request a resource
    # ─────────────────────────────────────────────────────────────────────────

    def request_resource(self):
        if not self.model.can_agent_request(self):
            self.frustration_counter += 1
            self.autonomy = PsychologyModel.update_autonomy(
                self.autonomy, "forced_decision"
            )
            return

        free_resources = self.model.get_free_resources()

        # scarcity perception: the less free resources, the higher
        total_resources = getattr(self.model, "num_resources", 10)
        scarcity = 1.0 - (len(free_resources) / max(1, total_resources))

        # subjective cost of cooperation: high autonomy cost reduces it
        base_cost = 1.0
        effective_cost = PsychologyModel.calculate_cooperation_cost(
            base_cost=base_cost,
            autonomy_felt=self.autonomy,
        )
        # if the effective cost is low, cooperation is easier → threshold is softened
        cost_factor = 1.0 - min(0.3, (base_cost - effective_cost))

        # cooperation probability (scaled by cost factor)
        p_coop = PsychologyModel.calculate_cooperation_probability(
            self.trust,
            self.satisfaction,
            self.autonomy,
            scarcity,
            self.weights,
        ) * cost_factor

        p_coop = self.model.adjust_cooperation_probability(self, p_coop)
        self.is_defecting = random.random() >= p_coop

        if free_resources:
            # free resource found → meaningful choice made
            resource = random.choice(free_resources)
            resource.is_occupied = True
            resource.user = self
            self.current_resource = resource
            self.usage_duration = 0

            # autonomy: a meaningful choice was made
            self.autonomy = PsychologyModel.update_autonomy(
                self.autonomy, "meaningful_choice"
            )
            # successful access → frustration is reset
            self.frustration_counter = 0

            if getattr(self.model, "verbose", False):
                print(
                    f"Agent {self.unique_id} ({self.agent_type}) acquired: "
                    f"{resource.unique_id}"
                )
            self.model.on_resource_acquired(self)
        else:
            # no resource → frustration counter increases
            self.frustration_counter += 1

            if self.frustration_counter >= FRUSTRATION_THRESHOLD:
                # system cannot provide resource: trust and autonomy decrease
                self.trust = PsychologyModel.update_trust(self.trust, "negative")
                self.autonomy = PsychologyModel.update_autonomy(
                    self.autonomy, "forced_decision"
                )
                self.frustration_counter = 0
            self.model.on_resource_unavailable(self)

    # ─────────────────────────────────────────────────────────────────────────
    # resource usage
    # ─────────────────────────────────────────────────────────────────────────

    def use_resource(self):
        """
        Manages the usage duration. Defectors hold the resource for a longer time.
        """
        self.usage_duration += 1

        limit = self.max_usage_duration
        if self.is_defecting:
            limit = limit * 2

        if self.usage_duration >= limit:
            self.release_resource()

    # ─────────────────────────────────────────────────────────────────────────
    # release the resource + feedback loop
    # ─────────────────────────────────────────────────────────────────────────

    def release_resource(self):
        """
        Releases the resource and triggers the psychological update loop:
        trust, satisfaction (DEA + equity bonus + procedural fairness) and autonomy.
        """
        if not self.current_resource:
            return

        if getattr(self.model, "verbose", False):
            print(
                f"Agent {self.unique_id} ({self.agent_type}) released: "
                f"{self.current_resource.unique_id}"
            )
        self.current_resource.is_occupied = False
        self.current_resource.user = None
        self.current_resource = None
        self.model.on_resource_released(self)

        # ── neighbor observation and equity perception ───────────────────────
        community_avg_wait = self.observe_neighbors()

        # equity bonus: compare own wait time to community average.
        # if below average, we feel fair → positive bonus
        # if above average, we feel unfair → negative bonus
        if community_avg_wait > 0:
            equity_ratio = self.wait_time / community_avg_wait
            # equity_ratio < 1 → kısa bekledik (iyi), > 1 → uzun bekledik (kötü)
            equity_bonus = max(-0.2, min(0.2, (1.0 - equity_ratio) * 0.2))
        else:
            equity_bonus = 0.0

        # procedural fairness: system bonus from the model + equity observation
        procedural_bonus = equity_bonus + self.model.procedural_bonus_modifier

        # ── similar to DEA satisfaction calculation ─────────────────────────
        inputs = self.wait_time + 1.0
        outputs = self.usage_duration

        new_sat = PsychologyModel.calculate_satisfaction(
            inputs_x=inputs,
            outputs_y=outputs,
            weights_v=1.0,
            weights_u=1.0,
            procedural_bonus=procedural_bonus,
        )

        # weighted moving average update
        self.satisfaction = (self.satisfaction * 0.7) + (new_sat * 0.3)

        # ── trust: successful use → positive experience ──────────────────────
        self.trust = PsychologyModel.update_trust(self.trust, "positive")

        # ── cumulative usage record (for Gini coefficient) ───────────────────
        self.cumulative_usage += self.usage_duration

        self.wait_time = 0

    # ─────────────────────────────────────────────────────────────────────────
    # neighbor observation (Equity Theory)
    # ─────────────────────────────────────────────────────────────────────────

    def observe_neighbors(self) -> float:
        """
        Calculates the average wait time of the neighboring PersonAgent's in the grid.
        Stores the result as perceived_community_fairness and returns it.

        Returns:
            float: the average wait time of the neighboring PersonAgent's (0 if no neighbor).
        """
        if not hasattr(self.model, "grid") or self.pos is None:
            return 0.0

        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=2
        )

        neighbor_wait_times = [
            n.wait_time
            for n in neighbors
            if isinstance(n, PersonAgent)
        ]

        if not neighbor_wait_times:
            return 0.0

        avg_wait = sum(neighbor_wait_times) / len(neighbor_wait_times)

        # 0-1 normalization: long wait = low fairness perception
        # 10 steps of wait is considered "bad"
        self.perceived_community_fairness = max(0.0, 1.0 - (avg_wait / 10.0))

        return avg_wait
