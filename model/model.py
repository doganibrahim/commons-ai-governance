from mesa import Model, DataCollector
from mesa.space import MultiGrid
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
    ):
        super().__init__()
        self.steps = 0
        self.num_agents = N_people
        self.num_resources = N_resources
        self.system_type = system_type
        self.procedural_bonus_modifier = procedural_bonus_modifier
        self.grid = MultiGrid(width, height, torus=False)
        self.running = True

        # Sürdürülebilirlik indeksi için kümülatif sayaçlar
        self.total_resource_idle_ticks = 0
        self.total_resource_ticks = 0

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

        # Sürdürülebilirlik sayaçlarını güncelle
        for a in self.agents:
            if isinstance(a, ResourceAgent):
                self.total_resource_ticks += 1
                if not a.is_occupied:
                    self.total_resource_idle_ticks += 1

        self.datacollector.collect(self)


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
