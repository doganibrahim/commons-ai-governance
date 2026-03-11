from mesa import Model
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
        # Simulation step counter
        self.steps = 0
        self.num_agents = N_people
        self.num_resources = N_resources
        # "baseline" | "ai_advisory" | "ai_autonomous" | "blockchain" | "integrated"
        self.system_type = system_type
        # Added directly to every agent's procedural bonus on resource release
        self.procedural_bonus_modifier = procedural_bonus_modifier
        # Initialize a grid where agents cannot wrap around edges
        self.grid = MultiGrid(width, height, torus=False)
        self.running = True

        dist = agent_type_distribution or AGENT_TYPE_DISTRIBUTION

        # Initialize and place Resource Agents
        for _ in range(self.num_resources):
            res = ResourceAgent(self)
            self.agents.add(res)

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(res, (x, y))

        # Build an ordered list of agent types according to the distribution
        agent_types = _build_agent_type_list(N_people, dist, self.random)

        # Initialize and place Person Agents
        for agent_type in agent_types:
            person = PersonAgent(self, agent_type=agent_type)
            self.agents.add(person)

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(person, (x, y))

    def get_free_resources(self):
        """Helper to return a list of resources that are not currently occupied."""
        resources = []
        for agent in self.agents:
            if isinstance(agent, ResourceAgent) and not agent.is_occupied:
                resources.append(agent)
        return resources

    def step(self):
        """Advance the model by one step, shuffling agent activation order."""
        self.steps += 1
        self.agents.shuffle_do("step")


def _build_agent_type_list(n: int, dist: dict, rng) -> list:
    """
    Returns a shuffled list of agent type strings of length n,
    proportioned according to the given distribution dict.
    Remaining slots (due to rounding) are filled with 'standard'.
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