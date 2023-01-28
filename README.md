# PyNeuralNet
A Python genetic Neural Network system

## Description

There isn't much to this repository, as it was only ever meant to be a test project.

I wanted to recreate the general idea of [another project of mine that I had worked on over a couple months](https://github.com/AyOhEe/NeuralNetworkExperiments) in Python, before moving on to expanding them elsewhere. I decided that I'd flesh out the repository a bit before I left though, just in case someone came across it and thought it was neat.

#### Networks, and how they operate

This project revolves around Genetic Neural Networks, primarily stored as collections of chromosome (.chr) files, describing aspects of their brains, such as their structure and how it's connected.

Each time input is fed to a network, it will "step" for a given number of times (`net.Step(inputs, n_steps=5)` for 5 steps, default is 1). Each step has each node sum its inputs, apply the activation function (defaults to sigmoid, can be set when creating a network), and store the result as its new value. This operation is done all at once for the entire network, as a simple matrix multiplication.

I designed the networks this way as I eventually want them to be able to develop a form of short term memory or sense of timing, and eventually form memories and alter their connections. Even currently, this method allows them to form reflexes in the form of short and responsive neural pathways, but also plan out action with longer and more complex neural pathways

## Usage
The [network.py](https://github.com/AyOhEe/PyNeuralNet//tree/main/network.py) file is standalone and only depends on NumPy, a common Python computation library. Simply drop it into any project you'd like to use it in and it'll work out of the box.

`Network` instances require a Genome to base themselves off of. An example of a few can be found in [Simulator/Genomes](https://github.com/AyOhEe/PyNeuralNet/tree/main/Simulator/Genomes/).
Genomes are composed of two files
  - `Nodes.chr` - This contains the biases of every node in the network, each one being represented as a 4 byte floating point number. Each node is stored as 4 bytes total.
  - `Connections.chr` - This contains each connection between any nodes in the network, each being represented as the index of the source and target nodes (both as 2 byte integers), and the weight of the connection as a 4 byte float. Each connection is stored as 8 bytes total.

## Future
In the future, I'd like to revisit and clean up the code here, especially the [example use](https://github.com/AyOhEe/PyNeuralNet/tree/main/Simulator/) I've given (currently [the Farama foundation's gymnasium](https://gymnasium.farama.org/#) and its [LunarLander environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/))

I do intend to come back to this project eventually, just to properly complete it, but it wouldn't be unlike me to simple not, instead just continuing whatever my current project is at the time. I do apologise if this ends up being the case.

## Notable mentions
#### Mr. David Randall Miller
This entire project, both this iteration and the project it was based off of, and my future endeavours in genetic artificial life/intelligence, have been inspired by [Mr. David Randall Miller on YouTube](https://www.youtube.com/@davidrandallmiller), with his [evolution simulator](https://www.youtube.com/watch?v=N3tRFayqVtk). <br>
If I had not seen his original video on his evolution simulator I am not sure that I would ever have done any of this. I thank him for both inspiring me and for wasting hours of my life rewatching that video many, many more times than is reasonable.

#### Mr. Dylan Cope
More recently, Mr. Dylan Cope's [Multicellular Microcosmos](https://www.youtube.com/watch?v=fEDqdvKO5Y0) appeared in my recommended feed on YouTube not many days after it was posted, and my head was immediately filled with ideas and inspiration, and was what drove me to start this project, such that I could attempt something akin to his. I thank him for sharing his endeavours with the internet, for he has inspired me greatly. You can find his work on [its respective GitHub repository](https://github.com/DylanCope/Evolving-Protozoa).

His work is something that I have wanted to create for some time now, and to see an idea similar to mine is incredible. His work is incredible and I highly recommend that you watch his video covering the amazing simulation he has designed.

Interestingly, he and I appear to have independently done our Neural Networks in a similar way. After reading it, I found that his code performs in a similar manner to my original project's code does(albeit his is much more elegant). The only difference between his code and *this* project is that I realised that these networks were just directed network graphs, easily representable as a sparse connection matrix, and chose to implement them as such.
