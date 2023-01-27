import matplotlib.pyplot as plt
import json
from .line import Line

class Graph:
    #creates a graph
    def __init__(self, title="Untitled Graph", x_label="X Axis", y_label="Y Axis", filename=None):
        #if there's a file, we should prioritise loading that
        if filename == None:
            self.title = title
            self.lines = []
            self.x_label = x_label
            self.y_label = y_label
            self.limits = None
        else:
            #open and load the file
            data = {}
            with open(filename) as f:
                data = json.loads(f.read())

            #if there is a title for the graph, use it
            if "Title" in data:
                self.title = data["Title"]
            else:
                self.title = title

            #if there are labels, use them
            if "Labels" in data:
                if len(data["Labels"]) == 2:
                    self.x_label = data["Labels"][0]
                    self.y_label = data["Labels"][1]
                else:
                    self.x_label = x_label
                    self.y_label = y_label

            #if there is valid limit data, use it
            if "xlim start" in data and "xlim end" in data and "ylim start" in data and "ylim end" in data:
                self.limits = [(data["xlim start"], data["xlim end"]), (data["ylim start"], data["ylim end"])]

            #for each of the lines in the data, construct and store the line
            self.lines = []
            if "Lines" in data:
                for line in data["Lines"]:
                    l = Line(line["Label"], **line["kwargs"])
                    l.add_points(zip(line["DataX"], line["DataY"]))
                    self.lines.append(l)

    #adds a line to this graph
    def add_line(self, line):
        self.lines.append(line)

    #displays this graph as a window on screen
    def display_graph(self):
        #clear the plot
        plt.clf()

        #calculate our x and y limits
        min_x = min([min(line.x_data) for line in self.lines])
        min_y = min([min(line.y_data) for line in self.lines])
        max_x = max([max(line.x_data) for line in self.lines])
        max_y = max([max(line.y_data) for line in self.lines])
       
        #set up our plot
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        #plot our lines
        for line in self.lines:
            plt.plot(line.x_data, line.y_data, label=line.label, **line.properties)
            
        #include a legend
        plt.legend()
        #show the plot
        plt.show()
        
        #leave the plot clean when we leave
        plt.clf()
    
    #saves this graph as a .png file
    def save_graph(self, filename):
        #clear the plot
        plt.clf()
        
        #calculate our x and y limits
        min_x = min([min(line.x_data) for line in self.lines])
        min_y = min([min(line.y_data) for line in self.lines])
        max_x = max([max(line.x_data) for line in self.lines])
        max_y = max([max(line.y_data) for line in self.lines])

        #set up our plot
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        #plot our lines
        for line in self.lines:
            plt.plot(line.x_data, line.y_data, label=line.label, **line.properties)
        
        #include a legend
        plt.legend()
        #save the graph
        plt.savefig(filename)

        #leave the plot clean when we leave
        plt.clf()

    #saves this graph's data as a .json file
    def save_graph_data(self, filename):
        #calculate our x and y limits
        min_x = min([min(line.x_data) for line in self.lines])
        min_y = min([min(line.y_data) for line in self.lines])
        max_x = max([max(line.x_data) for line in self.lines])
        max_y = max([max(line.y_data) for line in self.lines])

        #store this graph's data in a dictionary
        graph_data = {
            "Title": self.title, 
            "xlim start": min_x, 
            "xlim end": max_x, 
            "ylim start": min_y, 
            "ylim end": max_y,
            "Labels": [self.x_label, self.y_label],
            "Lines": [line.as_dict() for line in self.lines]
        }
        
        #convert the dictionary to json and save the json data to the file
        with open(filename, 'w+') as f:
            f.write(json.dumps(graph_data, indent=4))

    #reads in data from a json file and graphs and shows it using matplotlib
    @staticmethod
    def show_plot_file(filename):
        #clear the plot
        plt.clf()

        #open and load the file
        data = {}
        with open(filename) as f:
            data = json.loads(f.read())

        #if there is a title for the graph, use it
        if "Title" in data:
            plt.title(data["Title"])

        #if there are labels, use them
        if "Labels" in data:
            if len(data["Labels"]) == 2:
                plt.xlabel(data["Labels"][0])
                plt.ylabel(data["Labels"][1])

        #if there is valid limit data, use it
        if "xlim start" in data and "xlim end" in data and "ylim start" in data and "ylim end" in data:
            plt.xlim(data["xlim start"], data["xlim end"])
            plt.ylim(data["ylim start"], data["ylim end"])

        #for each of the lines in the data, plot the line
        if "Lines" in data:
            for line in data["Lines"]:
                # plot the points
                plt.plot(
                    line["DataX"], 
                    line["DataY"], 
                    label=line["Label"],
                    **line["kwargs"]
                )
                
        #include a legend
        plt.legend()
        #show the plot
        plt.show()

        #clear the plot
        plt.clf()

    #reads in data from a json file and graphs and saves it using matplotlib
    @staticmethod
    def save_plot_file(filename, outfilename):
        #clear the plot
        plt.clf()

        #open and load the file
        data = {}
        with open(filename) as f:
            data = json.loads(f.read())

        #if there is a title for the graph, use it
        if "Title" in data:
            plt.title(data["Title"])

        #if there are labels, use them
        if "Labels" in data:
            if len(data["Labels"]) == 2:
                plt.xlabel(data["Labels"][0])
                plt.ylabel(data["Labels"][1])

        #if there is valid limit data, use it
        if "xlim start" in data and "xlim end" in data and "ylim start" in data and "ylim end" in data:
            plt.xlim(data["xlim start"], data["xlim end"])
            plt.ylim(data["ylim start"], data["ylim end"])

        #for each of the lines in the data, plot the line
        if "Lines" in data:
            for line in data["Lines"]:
                # plot the points
                plt.plot(
                    line["DataX"], 
                    line["DataY"], 
                    label=line["Label"],
                    **line["kwargs"]
                )

        #include a legend
        plt.legend()
        #show the plot
        plt.savefig(outfilename)

        #clear the plot
        plt.clf()