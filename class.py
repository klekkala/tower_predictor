class Feature:
    def __init__(self, number):
        self.number = number
        
    def get_radius(self):
        if   self.number == 5:  return 64.0