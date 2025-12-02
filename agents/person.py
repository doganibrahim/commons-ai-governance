from mesa import Agent
import random

class PersonAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        #Currently used resource
        self.current_resource = None
        self.usage_duration = 0
        #Everyone can currently use the resource for a maximum of 5 steps.
        self.max_usage_duration = 5

    def step(self):
        if self.current_resource is None:
            self.request_resource()
        else:
            self.use_resource()

    def request_resource(self):
        #Find empty resources from the model
        free_resources = self.model.get_free_resources()

        if free_resources:
            #Select a resource at random
            resource = random.choice(free_resources)

            #Reserve the resource
            resource.is_occupied = True
            resource.user = self

            #Update self
            self.current_resource = resource
            self.usage_duration = 0
            print(f"Agent {self.unique_id} received the resource: {resource.unique_id}")

    def use_resource(self):
        self.usage_duration += 1
        #Is the time up?
        if self.usage_duration >= self.max_usage_duration:
            self.release_resource()

    def release_resource(self):
        if self.current_resource:
            print(f"Agent {self.unique_id} released the resource: {self.current_resource.unique_id}")
            self.current_resource.is_occupied = False
            self.current_resource.user = None
            self.current_resource = None