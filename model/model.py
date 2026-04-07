from mesa import Model, DataCollector
from mesa.space import MultiGrid
try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover
    IsolationForest = None
from agents.resource import ResourceAgent
from agents.person import PersonAgent

AGENT_TYPE_DISTRIBUTION = {
    "ideal":    0.25,
    "standard": 0.50,
    "toxic":    0.25,
}


class CommonsModel(Model):
    def __init__(
        self,
        N_people,
        N_resources,
        width,
        height,
        system_type: str = "baseline",
        procedural_bonus_modifier: float = 0.0,
        agent_type_distribution: dict = None,
        random_seed: int = None,
        verbose: bool = False,
    ):
        super().__init__(seed=random_seed)
        self.steps = 0
        self.num_agents = N_people
        self.num_resources = N_resources
        self.system_type = system_type
        self.procedural_bonus_modifier = procedural_bonus_modifier
        self.grid = MultiGrid(width, height, torus=False)
        self.running = True
        self.random_seed = random_seed
        self.verbose = verbose

        # Sürdürülebilirlik indeksi için kümülatif sayaçlar
        self.total_resource_idle_ticks = 0
        self.total_resource_ticks = 0
        self.total_conflicts = 0
        self.total_sanctions = 0
        self.last_detected_free_riders = 0
        self.agent_request_blocks = {}
        self.usage_ledger = {}
        self.detect_every_n_steps = 10
        self.anomaly_contamination = 0.15

        dist = agent_type_distribution or AGENT_TYPE_DISTRIBUTION

        # ── Resource Agents ─────────────────────────────────────────────────
        for _ in range(self.num_resources):
            res = ResourceAgent(self)
            self.agents.add(res)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(res, (x, y))

        # ── Person Agents ───────────────────────────────────────────────────
        agent_types = _build_agent_type_list(N_people, dist, self.random)
        for agent_type in agent_types:
            person = PersonAgent(self, agent_type=agent_type)
            self.agents.add(person)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(person, (x, y))
            self.agent_request_blocks[person.unique_id] = 0
            self.usage_ledger[person.unique_id] = {
                "acquired": 0,
                "released": 0,
                "blocked": 0,
                "iforest_anomaly": 0,
            }

        # ── DataCollector ───────────────────────────────────────────────────
        self.datacollector = DataCollector(
            model_reporters={
                "mean_trust":           _mean_trust,
                "mean_autonomy":        _mean_autonomy,
                "mean_satisfaction":    _mean_satisfaction,
                "mean_community_fairness": _mean_community_fairness,
                "cooperation_rate":     _cooperation_rate,
                "free_rider_ratio":     _free_rider_ratio,
                "resource_utilization": _resource_utilization,
                "gini_coefficient":     _gini_coefficient,
                "sustainability_index": _sustainability_index,
                "conflict_rate":        _conflict_rate,
                "sanction_rate":        _sanction_rate,
                "iforest_anomaly_ratio": _iforest_anomaly_ratio,
                "system_type":          lambda m: m.system_type,
            },
            agent_reporters={
                "trust":       "trust",
                "autonomy":    "autonomy",
                "satisfaction": "satisfaction",
                "is_defecting": "is_defecting",
                "wait_time":   "wait_time",
                "usage_duration": "usage_duration",
                "agent_type":  "agent_type",
                "perceived_community_fairness": "perceived_community_fairness",
                "last_iforest_label": "last_iforest_label",
            },
            agenttype_reporters={
                PersonAgent: {
                    "trust":       "trust",
                    "autonomy":    "autonomy",
                    "satisfaction": "satisfaction",
                },
            },
        )

    def get_free_resources(self):
        """Helper to return a list of resources that are not currently occupied."""
        return [
            a for a in self.agents
            if isinstance(a, ResourceAgent) and not a.is_occupied
        ]

    def step(self):
        """Advance the model by one step, shuffling agent activation order."""
        self.steps += 1
        self.agents.shuffle_do("step")
        if self.steps % self.detect_every_n_steps == 0:
            self.run_iforest_detection()

        # Sürdürülebilirlik sayaçlarını güncelle
        for a in self.agents:
            if isinstance(a, ResourceAgent):
                self.total_resource_ticks += 1
                if not a.is_occupied:
                    self.total_resource_idle_ticks += 1

        self.datacollector.collect(self)

    def can_agent_request(self, agent):
        if self.agent_request_blocks.get(agent.unique_id, 0) > self.steps:
            self.usage_ledger[agent.unique_id]["blocked"] += 1
            return False
        return True

    def adjust_cooperation_probability(self, agent, p_coop):
        adjusted = p_coop
        if self.system_type == "ai_advisory":
            if agent.last_iforest_label == -1:
                adjusted = min(1.0, adjusted + 0.10)
        elif self.system_type == "ai_autonomous":
            if agent.last_iforest_label == -1:
                adjusted = min(1.0, adjusted + 0.15)
        elif self.system_type == "blockchain_partial":
            adjusted = min(1.0, adjusted + 0.04)
        elif self.system_type == "blockchain_full":
            adjusted = min(1.0, adjusted + 0.08)
        elif self.system_type == "integrated":
            bonus = 0.08 + (0.08 if agent.last_iforest_label == -1 else 0.0)
            adjusted = min(1.0, adjusted + bonus)
        return max(0.0, adjusted)

    def on_resource_acquired(self, agent):
        self.usage_ledger[agent.unique_id]["acquired"] += 1

    def on_resource_released(self, agent):
        self.usage_ledger[agent.unique_id]["released"] += 1

    def on_resource_unavailable(self, _agent):
        self.total_conflicts += 1

    def run_iforest_detection(self):
        persons = _get_person_agents(self)
        if len(persons) < 4:
            return
        if IsolationForest is None:
            for agent in persons:
                agent.last_iforest_label = 1
            self.last_detected_free_riders = 0
            return
        features = []
        for a in persons:
            features.append(
                [
                    a.cumulative_usage,
                    a.wait_time,
                    1 if a.is_defecting else 0,
                    a.autonomy,
                    a.satisfaction,
                    self.usage_ledger[a.unique_id]["blocked"],
                ]
            )
        detector = IsolationForest(
            contamination=self.anomaly_contamination,
            random_state=self.random_seed if self.random_seed is not None else 42,
        )
        labels = detector.fit_predict(features)
        anomaly_count = 0
        for agent, label in zip(persons, labels):
            agent.last_iforest_label = label
            if label == -1:
                anomaly_count += 1
                self.usage_ledger[agent.unique_id]["iforest_anomaly"] += 1
                if self.system_type in ("ai_autonomous", "integrated"):
                    block_until = self.steps + 3
                    self.agent_request_blocks[agent.unique_id] = max(
                        self.agent_request_blocks[agent.unique_id], block_until
                    )
                    self.total_sanctions += 1
        self.last_detected_free_riders = anomaly_count


# ═════════════════════════════════════════════════════════════════════════════
# Yardımcı fonksiyonlar (DataCollector model_reporters için)
# ═════════════════════════════════════════════════════════════════════════════

def _get_person_agents(model):
    return [a for a in model.agents if isinstance(a, PersonAgent)]


def _mean_trust(model):
    persons = _get_person_agents(model)
    if not persons:
        return 0.0
    return sum(a.trust for a in persons) / len(persons)


def _mean_autonomy(model):
    persons = _get_person_agents(model)
    if not persons:
        return 0.0
    return sum(a.autonomy for a in persons) / len(persons)


def _mean_satisfaction(model):
    persons = _get_person_agents(model)
    if not persons:
        return 0.0
    return sum(a.satisfaction for a in persons) / len(persons)


def _mean_community_fairness(model):
    persons = _get_person_agents(model)
    if not persons:
        return 0.0
    return sum(a.perceived_community_fairness for a in persons) / len(persons)


def _cooperation_rate(model):
    persons = _get_person_agents(model)
    if not persons:
        return 0.0
    cooperators = sum(1 for a in persons if not a.is_defecting)
    return cooperators / len(persons)


def _free_rider_ratio(model):
    persons = _get_person_agents(model)
    if not persons:
        return 0.0
    defectors = sum(1 for a in persons if a.is_defecting)
    return defectors / len(persons)


def _resource_utilization(model):
    resources = [a for a in model.agents if isinstance(a, ResourceAgent)]
    if not resources:
        return 0.0
    occupied = sum(1 for r in resources if r.is_occupied)
    return occupied / len(resources)


def _gini_coefficient(model):
    """
    Kümülatif kaynak kullanım sürelerinden Gini katsayısı hesaplar.
    0 = tam eşitlik, 1 = tam eşitsizlik.
    """
    persons = _get_person_agents(model)
    usages = sorted(a.cumulative_usage for a in persons)
    n = len(usages)
    if n == 0 or sum(usages) == 0:
        return 0.0
    total = sum(usages)
    cumulative_sum = sum(i * y for i, y in enumerate(usages, 1))
    return (2 * cumulative_sum) / (n * total) - (n + 1) / n


def _sustainability_index(model):
    """
    Kaynakların boş kalma oranı (kümülatif).
    Yüksek = kaynaklar tükenmeden paylaşılıyor → sürdürülebilir.
    """
    if model.total_resource_ticks == 0:
        return 1.0
    return model.total_resource_idle_ticks / model.total_resource_ticks


def _conflict_rate(model):
    if model.steps == 0:
        return 0.0
    return model.total_conflicts / model.steps


def _sanction_rate(model):
    if model.steps == 0:
        return 0.0
    return model.total_sanctions / model.steps


def _iforest_anomaly_ratio(model):
    persons = _get_person_agents(model)
    if not persons:
        return 0.0
    return model.last_detected_free_riders / len(persons)


# ═════════════════════════════════════════════════════════════════════════════
# Ajan tipi dağılımı oluşturucu
# ═════════════════════════════════════════════════════════════════════════════

def _build_agent_type_list(n: int, dist: dict, rng) -> list:
    """
    Returns a shuffled list of agent type strings of length n,
    proportioned according to the given distribution dict.
    """
    types = []
    for agent_type, ratio in dist.items():
        count = round(n * ratio)
        types.extend([agent_type] * count)

    while len(types) < n:
        types.append("standard")
    types = types[:n]

    rng.shuffle(types)
    return types
