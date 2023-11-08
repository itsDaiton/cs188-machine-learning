import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # Výpočet skalárního součinu vstupních neuronů a jejich vah
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # Klasifikace vstupního neuronu x - pokud je výsledek skalárního součinu kladný, tak 1, jinak -1
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # Trénování perceptronu - pro vstupní data která se špatně klasifikují se upravují váhy dokud se všechna data neklasifikují správně
        missclassified = True
        while missclassified:
            missclassified = False
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    missclassified = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Propojení vstupní vrstvy s první skrytou vrstvou
        self.w1 = nn.Parameter(1, 200)
        self.b1 = nn.Parameter(1, 200)
        
        # Propojení první skryté vrstvy s druhou skrytou vrstvou
        self.w2 = nn.Parameter(200, 100)
        self.b2 = nn.Parameter(1, 100)
        
        # Propojení druhé skryté vrstvy s výstupní vrstvou
        self.w3 = nn.Parameter(100, 1)
        self.b3 = nn.Parameter(1, 1)
        
        # Seskupení všech vah a biasů do jedné proměnné
        self.hyperparameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        
        # Zbylé hyperparametry - velikost dávky a rychlost učení
        self.batch_size = 100
        self.learning_rate = 0.02

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Výpočet výstupní aktivační hodnoty pro první skrytou vrstvu pomocí aktivační funkce ReLU
        output1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        
        # Výpočet výstupní aktivační hodnoty pro druhou skrytou vrstvu pomocí aktivační funkce ReLU
        output2 = nn.ReLU(nn.AddBias(nn.Linear(output1, self.w2), self.b2))
        
        # Výpočet výstupní aktivační hodnoty pro výstupní vrstvu
        # Zde se již funkce ReLU nepoužívá, protože chceme na výstupu i záporné hodnoty, které ReLU ořezává
        output3 = nn.AddBias(nn.Linear(output2, self.w3), self.b3)
        
        # Výsledná predikce modelu
        return output3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Výpočet ztrátové funkce - rozdíl mezi získanou hodnotou a předpokládanou hodnotou
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Prvotní incilializace ztrátové funkce na nekonečno, jelikož chceme v rámci trénování minimalizovat tuto hodnotu
        loss = float('inf')
        # Trénování modelu - dokud není ztrátová funkce menší než 0.02, tak se bude model neustále trénovat
        while loss > 0.02:
            # Iterace přes dávku dat z datasetu
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # Výpočet gradientů pro jednotlivé hyperparametry
                gradients = nn.gradients(loss, self.hyperparameters)
                # Převod ztrátové funkce zpět na skalární hodnotu
                loss = nn.as_scalar(loss)
                # Backpropagation - zpětná aktualizace vah a biasů pro celou síť
                for i in range(len(gradients)):
                    self.hyperparameters[i].update(gradients[i], -self.learning_rate)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Propojení vstupní vrstvy (28*28 pixelů) s první skrytou vrstvou
        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        
        # Propojení první skryté vrstvy s druhou skrytou vrstvou
        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        
        # Propojení druhé skryté vrstvy s výstupní vrstvou
        self.w3 = nn.Parameter(128, 64)
        self.b3 = nn.Parameter(1, 64)
        
        # Propojení třetí skryté vrstvy s výstupní vrstvou (10 tříd)
        self.w4 = nn.Parameter(64, 10)
        self.b4 = nn.Parameter(1, 10)
        
        # Seskupení všech vah a biasů do jedné proměnné
        self.hyperparameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]
        
        # Zbylé hyperparametry - velikost dávky a rychlost učení
        self.batch_size = 40
        self.learning_rate = 0.1

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Výpočet výstupní aktivační hodnoty pro první skrytou vrstvu pomocí aktivační funkce ReLU
        output1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        
        # Výpočet výstupní aktivační hodnoty pro druhou skrytou vrstvu pomocí aktivační funkce ReLU
        output2 = nn.ReLU(nn.AddBias(nn.Linear(output1, self.w2), self.b2))
        
        # Výpočet výstupní aktivační hodnoty pro třetí skrytou vrstvu pomocí aktivační funkce ReLU
        output3 = nn.ReLU(nn.AddBias(nn.Linear(output2, self.w3), self.b3))
        
        # Výpočet výstupní aktivační hodnoty pro výstupní vrstvu, zde bez ReLU (dle zadání)
        output4 = nn.AddBias(nn.Linear(output3, self.w4), self.b4)
        
        # Výsledná predikce modelu
        return output4

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Výpočet ztrátové funkce pro vícetřídní klasifikaci
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"     
        validation_accuracy = 0     
        # Trénování modelu - dokud není přesnost modelu na validačních datech větší než 0.98, tak se bude model neustále trénovat
        while validation_accuracy <= 0.98:
            # Iterace přes dávku dat z datasetu
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # Výpočet gradientů pro jednotlivé hyperparametry
                gradients = nn.gradients(loss, self.hyperparameters)
                # Backpropagation - zpětná aktualizace vah a biasů pro celou síť
                for i in range(len(gradients)):
                    self.hyperparameters[i].update(gradients[i], -self.learning_rate)
            # Výpočet přesnosti modelu na validačních datech
            validation_accuracy = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        
        # Propojení vstupní vrstvy (47 znaků) s první skrytou vrstvou (zpracování prvního písmena slova)
        self.w1 = nn.Parameter(self.num_chars, 256)
        self.b1 = nn.Parameter(1, 256)
        
        # Kekurzivní skrytá podsíť pro zpracování dalších písmen slova (tento kód se rekurzivně volá v cyklu níže)
        self.w2 = nn.Parameter(256, 256)
        self.b2 = nn.Parameter(1, 256)
        
        # Poslední vrstva pro vytvoření predikce modelu (klasifikace do 5 jazyků)
        self.w3 = nn.Parameter(256, len(self.languages))
        self.b3 = nn.Parameter(1, len(self.languages))
        
        # Seskupení všech vah a biasů do jedné proměnné
        self.hyperparameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        
        # Zbylé hyperparametry - velikost dávky a rychlost učení
        self.batch_size = 100
        self.learning_rate = 0.15

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Výpočet výstupní aktivační hodnoty pro první písmeno slova (stejná architekrura jako v předchozích dopředných sítích)
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w1), self.b1))
        
        # Dále musíme spojit výstup předchozího kroku s dalším písmenem slova (napojení na skrytou podsíť)
        # Výsledek je vektor prvních dvou písmen slova
        # Tento proces poté opakujeme pro všechna písmena slova, dokud nezpracujeme celé slovo (rekuzrivní volání v cyklu)
        # Důležité je, že výstup z předchozího kroku je vždy vstupem pro další krok (skrytý stav [h])
        for x in xs[1:]:
            h = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(x, self.w1), nn.Linear(h, self.w2)), self.b1))
        # Výsledná predikce modelu
        return nn.AddBias(nn.Linear(h, self.w3), self.b3)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Výpočet ztrátové funkce - opět používáme SoftmaxLoss, jelikož se jedná o vícetřídní klasifikaci
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"  
        # Trénování modelu - 20 epoch trénování
        # V zadání totiž autoři píší, že jejich refereční model dosahuje přesnosti cca. 0.89 na validačních datech po 10 - 20 epochách
        # Trénování tedy neskončí dříve jako u předchozích modelů, ale vždy až po 20 epochách
        # Po skončení trénování se vypíše přesnost modelu na validačních datech a autograder i tak vyhodnotí úspěšnost
        for i in range(20):
            # Iterace přes dávku dat z datasetu
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # Výpočet gradientů pro jednotlivé hyperparametry
                gradients = nn.gradients(loss, self.hyperparameters)
                # Backpropagation - zpětná aktualizace vah a biasů pro celou síť
                for i in range(len(gradients)):
                    self.hyperparameters[i].update(gradients[i], -self.learning_rate)
