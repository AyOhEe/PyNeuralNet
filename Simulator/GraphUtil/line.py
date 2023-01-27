class Line:
    #creates a line
    def __init__(self, label="", **kwargs):
        self.label = label
        self.x_data = []
        self.y_data = []
        self.properties = kwargs

    #adds a point to the line
    def add_point(self, x, y):
        self.x_data.append(x)
        self.y_data.append(y)

    #adds several points to the line. expects list of tuples!
    def add_points(self, tuple_list):
        for tup in tuple_list:
            self.x_data.append(tup[0])
            self.y_data.append(tup[1])
        
    #returns this line as a dictionary
    def as_dict(self):
        return { "Label": self.label, "DataX": self.x_data, "DataY": self.y_data, "kwargs": self.properties}