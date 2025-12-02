from mesa import Agent


class ResourceAgent(Agent):
    def __init__(self, model):
        # Mesa 3+ için imza: Agent(model, *args, **kwargs)
        super().__init__(model)
        # Is resource occupied?
        self.is_occupied = False
        # Who is currently using the resource?
        self.user = None

    def step(self):
        # Resources are passive, they cannot make decisions on their own.
        pass