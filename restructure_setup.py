class Simulation:
    """
    Docstring here
    
    """

    def __init__(self, density, temp, N_particles=108, L=10, dim=3):
        self.density = density
        #etc
        pass

    def __repr__(self):
        # write a formal representation for this class
        pass

    def __str__(self):
        # write an informal string representation for this class
        pass

    def equilibrate(self):
        # see lecture 4 and coding guidelines
        pass

    def simulate(self, algorithm, time):
        # see lecture 4
        # time how long to simulate?
        # algorithm: euler or verlet

        #i'd say: based on chosen algorithm, import correct function
        # from other file
        pass

    def quickshow(self):
        # just an idea
        pass

    def save(self):
        # and specify in what form
        pass


# also maybe: class to store state

class State_of_Simulation(Simulation):
    # inheritance of Simulation used here