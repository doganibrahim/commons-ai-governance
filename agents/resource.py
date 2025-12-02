from mesa import Agent

class ResourceAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        #Is resource occupied?
        self.is_occupied = False
        #Who is currently using the resource?
        self.user = None

    def step(self):
        #Resources are passive, they cannot make decisions on their own.
        pass