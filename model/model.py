from mesa import Model
from mesa.space import MultiGrid
from agents.resource import ResourceAgent
from agents.person import PersonAgent

class CommonsModel(Model):
    def __init__(self, N_people, N_resources, width, height):
        super().__init__()
        # Simulation step counter
        self.steps = 0
        self.num_agents = N_people
        self.num_resources = N_resources
        # Initialize a grid where agents cannot wrap around edges
        self.grid = MultiGrid(width, height, torus=False)
        self.running = True

        # Initialize and place Resource Agents
        for _ in range(self.num_resources):
            res = ResourceAgent(self)
            # Add to the model's agent registry
            self.agents.add(res)

            # Assign random coordinates
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(res, (x, y))

        # Initialize and place Person Agents
        for _ in range(self.num_agents):
            person = PersonAgent(self)
            self.agents.add(person)

            # Assign random coordinates
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
        # Increase the step counter
        self.steps += 1
        self.agents.shuffle_do("step")